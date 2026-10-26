/****************************************************************************
 *
 * MODULE:       i.delineate.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Minimal reader for PyTorch checkpoints (torch.save zip
 *               format): parses the zip central directory and the
 *               pickle of a state_dict or of a whole model, without any
 *               Python dependency.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "progress.h"
#include "pth_loader.h"

/* Zip archive handling. torch.save() writes stored (uncompressed)
 * entries, so an entry's payload can be read in place once the local
 * header has been skipped. */

struct zip_entry {
    char *name;
    int method;
    uint64_t size;
    uint64_t local_offset;
};

struct zip_dir {
    int n;
    struct zip_entry *e;
};

static uint16_t rd16(const unsigned char *p)
{
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const unsigned char *p)
{
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

static void read_at(FILE *fp, off_t off, void *buf, size_t n, const char *path)
{
    if (fseeko(fp, off, SEEK_SET) != 0 || fread(buf, 1, n, fp) != n)
        G_fatal_error(_("Unable to read %lu bytes at offset %lld in <%s>"),
                      (unsigned long)n, (long long)off, path);
    progress_bytes(n);
}

static void zip_read_dir(FILE *fp, const char *path, struct zip_dir *zd)
{
    unsigned char *tail, *cd, *p;
    off_t fsize, tail_off, eocd;
    size_t tail_len;
    uint64_t cd_off, cd_size, nent;
    long i;

    if (fseeko(fp, 0, SEEK_END) != 0)
        G_fatal_error(_("Unable to seek in <%s>"), path);
    fsize = ftello(fp);
    tail_len = fsize < 65557 ? (size_t)fsize : 65557;
    tail_off = fsize - (off_t)tail_len;
    tail = G_malloc(tail_len);
    read_at(fp, tail_off, tail, tail_len, path);

    eocd = -1;
    for (i = (long)tail_len - 22; i >= 0; i--) {
        if (rd32(tail + i) == 0x06054b50) {
            eocd = i;
            break;
        }
    }
    if (eocd < 0)
        G_fatal_error(_("<%s> is not a zip archive (PyTorch >= 1.6 "
                        "checkpoint format expected)"),
                      path);

    nent = rd16(tail + eocd + 10);
    cd_size = rd32(tail + eocd + 12);
    cd_off = rd32(tail + eocd + 16);

    if (nent == 0xFFFF || cd_size == 0xFFFFFFFF || cd_off == 0xFFFFFFFF) {
        /* Zip64: the locator sits right before the classic record. */
        unsigned char loc[20], rec[56];

        if (eocd < 20 || rd32(tail + eocd - 20) != 0x07064b50)
            G_fatal_error(_("Corrupt zip64 directory in <%s>"), path);
        memcpy(loc, tail + eocd - 20, 20);
        read_at(fp, (off_t)rd64(loc + 8), rec, sizeof(rec), path);
        if (rd32(rec) != 0x06064b50)
            G_fatal_error(_("Corrupt zip64 end record in <%s>"), path);
        nent = rd64(rec + 32);
        cd_size = rd64(rec + 40);
        cd_off = rd64(rec + 48);
    }
    G_free(tail);

    cd = G_malloc(cd_size);
    read_at(fp, (off_t)cd_off, cd, cd_size, path);

    zd->n = (int)nent;
    zd->e = G_calloc(nent, sizeof(struct zip_entry));
    p = cd;
    for (i = 0; i < (long)nent; i++) {
        struct zip_entry *e = &zd->e[i];
        unsigned nlen, elen, clen;
        uint32_t csize, usize, loff;
        const unsigned char *x, *xend;

        if (p + 46 > cd + cd_size || rd32(p) != 0x02014b50)
            G_fatal_error(_("Corrupt zip central directory in <%s>"), path);
        e->method = rd16(p + 10);
        csize = rd32(p + 20);
        usize = rd32(p + 24);
        nlen = rd16(p + 28);
        elen = rd16(p + 30);
        clen = rd16(p + 32);
        loff = rd32(p + 42);
        e->name = G_malloc(nlen + 1);
        memcpy(e->name, p + 46, nlen);
        e->name[nlen] = '\0';
        e->size = usize;
        e->local_offset = loff;

        /* Zip64 extra field carries the 64-bit values that overflowed. */
        x = p + 46 + nlen;
        xend = x + elen;
        while (x + 4 <= xend) {
            unsigned id = rd16(x), len = rd16(x + 2);
            const unsigned char *v = x + 4;

            if (id == 0x0001) {
                if (usize == 0xFFFFFFFF) {
                    e->size = rd64(v);
                    v += 8;
                }
                if (csize == 0xFFFFFFFF)
                    v += 8;
                if (loff == 0xFFFFFFFF)
                    e->local_offset = rd64(v);
            }
            x += 4 + len;
        }
        p += 46 + nlen + elen + clen;
    }
    G_free(cd);
}

static const struct zip_entry *zip_find(const struct zip_dir *zd,
                                        const char *name)
{
    int i;

    for (i = 0; i < zd->n; i++)
        if (strcmp(zd->e[i].name, name) == 0)
            return &zd->e[i];
    return NULL;
}

static off_t zip_data_offset(FILE *fp, const char *path,
                             const struct zip_entry *e)
{
    unsigned char h[30];

    if (e->method != 0)
        G_fatal_error(_("Zip entry <%s> in <%s> is compressed; only stored "
                        "entries (as written by torch.save) are supported"),
                      e->name, path);
    read_at(fp, (off_t)e->local_offset, h, sizeof(h), path);
    if (rd32(h) != 0x04034b50)
        G_fatal_error(_("Corrupt local header for <%s> in <%s>"), e->name,
                      path);
    return (off_t)e->local_offset + 30 + rd16(h + 26) + rd16(h + 28);
}

/* Pickle virtual machine, restricted to what torch.save() emits for a
 * state_dict or a whole model: dicts/OrderedDicts of tensors rebuilt
 * from persistent storage references, and objects built with NEWOBJ or
 * REDUCE then BUILD. No class is ever called: an object only records
 * its class and the state given to BUILD. Unknown opcodes are fatal. */

enum pv_type {
    PV_NONE,
    PV_BOOL,
    PV_INT,
    PV_FLOAT,
    PV_STR,
    PV_TUPLE,
    PV_LIST,
    PV_DICT,
    PV_GLOBAL,
    PV_STORAGE,
    PV_TENSOR,
    PV_MARK,
    PV_OBJ
};

struct pv {
    enum pv_type t;
    long i;
    double f;
    char *s;  /* PV_STR text, PV_GLOBAL module, PV_STORAGE key. */
    char *s2; /* PV_GLOBAL name. */
    int n, cap;
    struct pv **items; /* Tuple/list items; dict as k0, v0, k1, v1... */
    enum pth_dtype dtype;
    size_t esize;     /* PV_STORAGE element size in bytes. */
    struct pv *cls;   /* PV_OBJ class (a PV_GLOBAL). */
    struct pv *state; /* PV_OBJ state given to BUILD, or NULL. */
    struct pv *storage;
    int ndim;
    long shape[PTH_MAX_DIMS], stride[PTH_MAX_DIMS];
};

struct pvm {
    const unsigned char *p, *end;
    struct pv **stack;
    int sp, scap;
    struct pv **memo;
    long mcap;
    struct pv **all; /* Every allocated value, for cleanup. */
    long nall, allcap;
};

static struct pv *pv_new(struct pvm *m, enum pv_type t)
{
    struct pv *v = G_calloc(1, sizeof(struct pv));

    v->t = t;
    if (m->nall == m->allcap) {
        m->allcap = m->allcap ? 2 * m->allcap : 4096;
        m->all = G_realloc(m->all, m->allcap * sizeof(struct pv *));
    }
    m->all[m->nall++] = v;
    return v;
}

static void pv_append(struct pv *c, struct pv *v)
{
    if (c->n == c->cap) {
        c->cap = c->cap ? 2 * c->cap : 8;
        c->items = G_realloc(c->items, c->cap * sizeof(struct pv *));
    }
    c->items[c->n++] = v;
}

static void push(struct pvm *m, struct pv *v)
{
    if (m->sp == m->scap) {
        m->scap = m->scap ? 2 * m->scap : 256;
        m->stack = G_realloc(m->stack, m->scap * sizeof(struct pv *));
    }
    m->stack[m->sp++] = v;
}

static struct pv *pop(struct pvm *m)
{
    if (m->sp <= 0)
        G_fatal_error(_("Pickle stack underflow in checkpoint"));
    return m->stack[--m->sp];
}

static struct pv *top(struct pvm *m)
{
    if (m->sp <= 0)
        G_fatal_error(_("Pickle stack underflow in checkpoint"));
    return m->stack[m->sp - 1];
}

/* Return the stack index of the topmost mark and discard the mark. */
static int pop_mark(struct pvm *m)
{
    int k;

    for (k = m->sp - 1; k >= 0; k--)
        if (m->stack[k]->t == PV_MARK)
            return k;
    G_fatal_error(_("Pickle mark not found in checkpoint"));
    return -1;
}

static const unsigned char *take(struct pvm *m, size_t n)
{
    const unsigned char *q = m->p;

    if ((size_t)(m->end - m->p) < n)
        G_fatal_error(_("Truncated pickle in checkpoint"));
    m->p += n;
    return q;
}

static char *take_line(struct pvm *m)
{
    const unsigned char *q = m->p;
    char *s;

    while (m->p < m->end && *m->p != '\n')
        m->p++;
    if (m->p >= m->end)
        G_fatal_error(_("Truncated pickle in checkpoint"));
    s = G_malloc(m->p - q + 1);
    memcpy(s, q, m->p - q);
    s[m->p - q] = '\0';
    m->p++;
    return s;
}

static void memo_put(struct pvm *m, long idx, struct pv *v)
{
    if (idx < 0)
        G_fatal_error(_("Invalid pickle memo index"));
    if (idx >= m->mcap) {
        long ncap = m->mcap ? m->mcap : 1024;

        while (ncap <= idx)
            ncap *= 2;
        m->memo = G_realloc(m->memo, ncap * sizeof(struct pv *));
        memset(m->memo + m->mcap, 0, (ncap - m->mcap) * sizeof(struct pv *));
        m->mcap = ncap;
    }
    m->memo[idx] = v;
}

static struct pv *memo_get(struct pvm *m, long idx)
{
    if (idx < 0 || idx >= m->mcap || !m->memo[idx])
        G_fatal_error(_("Invalid pickle memo reference %ld"), idx);
    return m->memo[idx];
}

static struct pv *make_str(struct pvm *m, const unsigned char *s, size_t n)
{
    struct pv *v = pv_new(m, PV_STR);

    v->s = G_malloc(n + 1);
    memcpy(v->s, s, n);
    v->s[n] = '\0';
    return v;
}

static struct pv *make_tuple_from(struct pvm *m, int from)
{
    struct pv *v = pv_new(m, PV_TUPLE);
    int k;

    for (k = from; k < m->sp; k++)
        pv_append(v, m->stack[k]);
    m->sp = from;
    return v;
}

static long pv_int(const struct pv *v)
{
    if (v->t == PV_INT || v->t == PV_BOOL)
        return v->i;
    G_fatal_error(_("Integer expected in checkpoint pickle"));
    return 0;
}

static int is_global(const struct pv *v, const char *mod, const char *name)
{
    return v->t == PV_GLOBAL && strcmp(v->s, mod) == 0 &&
           strcmp(v->s2, name) == 0;
}

static struct pv *reduce(struct pvm *m, struct pv *fn, struct pv *args)
{
    struct pv *v;
    int d;

    if (args->t != PV_TUPLE)
        G_fatal_error(_("Pickle REDUCE without argument tuple"));

    if (is_global(fn, "torch._utils", "_rebuild_tensor_v2") ||
        is_global(fn, "torch._utils", "_rebuild_tensor")) {
        struct pv *size, *stride;

        if (args->n < 4 || args->items[0]->t != PV_STORAGE ||
            args->items[2]->t != PV_TUPLE || args->items[3]->t != PV_TUPLE)
            G_fatal_error(_("Unexpected tensor rebuild arguments"));
        size = args->items[2];
        stride = args->items[3];
        if (size->n > PTH_MAX_DIMS || size->n != stride->n)
            G_fatal_error(_("Unsupported tensor rank %d"), size->n);
        v = pv_new(m, PV_TENSOR);
        v->storage = args->items[0];
        v->i = pv_int(args->items[1]);
        v->ndim = size->n;
        for (d = 0; d < size->n; d++) {
            v->shape[d] = pv_int(size->items[d]);
            v->stride[d] = pv_int(stride->items[d]);
        }
        return v;
    }
    if (is_global(fn, "torch._utils", "_rebuild_parameter") ||
        is_global(fn, "torch._utils", "_rebuild_parameter_with_state")) {
        if (args->n < 1 || args->items[0]->t != PV_TENSOR)
            G_fatal_error(_("Unexpected parameter rebuild arguments"));
        return args->items[0];
    }
    if (is_global(fn, "collections", "OrderedDict"))
        return pv_new(m, PV_DICT);

    v = pv_new(m, PV_OBJ);
    v->cls = fn;
    return v;
}

static struct pv *persistent_load(struct pvm *m, struct pv *pid)
{
    struct pv *v, *type;
    const char *tn;

    if (pid->t != PV_TUPLE || pid->n < 5 || pid->items[0]->t != PV_STR ||
        strcmp(pid->items[0]->s, "storage") != 0 ||
        pid->items[1]->t != PV_GLOBAL || pid->items[2]->t != PV_STR)
        G_fatal_error(_("Unsupported persistent id in checkpoint"));

    type = pid->items[1];
    tn = type->s2;
    v = pv_new(m, PV_STORAGE);
    v->esize = 2;
    if (strcmp(tn, "FloatStorage") == 0) {
        v->dtype = PTH_F32;
        v->esize = 4;
    }
    else if (strcmp(tn, "HalfStorage") == 0)
        v->dtype = PTH_F16;
    else if (strcmp(tn, "BFloat16Storage") == 0)
        v->dtype = PTH_BF16;
    else {
        /* Integer and boolean buffers (e.g. BatchNorm's
         * num_batches_tracked) are not indexed. */
        v->dtype = PTH_OTHER;
        if (strcmp(tn, "LongStorage") == 0 || strcmp(tn, "DoubleStorage") == 0)
            v->esize = 8;
        else if (strcmp(tn, "IntStorage") == 0)
            v->esize = 4;
        else if (strcmp(tn, "ShortStorage") == 0)
            v->esize = 2;
        else
            v->esize = 1;
    }
    v->s = pid->items[2]->s;
    return v;
}

static struct pv *run_pickle(struct pvm *m)
{
    for (;;) {
        unsigned char op = *take(m, 1);
        struct pv *v, *a, *b;
        int k;

        switch (op) {
        case 0x80: /* PROTO */
            take(m, 1);
            break;
        case 0x95: /* FRAME */
            take(m, 8);
            break;
        case '.': /* STOP */
            return pop(m);
        case '(': /* MARK */
            push(m, pv_new(m, PV_MARK));
            break;
        case '}': /* EMPTY_DICT */
            push(m, pv_new(m, PV_DICT));
            break;
        case ']': /* EMPTY_LIST */
            push(m, pv_new(m, PV_LIST));
            break;
        case ')': /* EMPTY_TUPLE */
            push(m, pv_new(m, PV_TUPLE));
            break;
        case 'N': /* NONE */
            push(m, pv_new(m, PV_NONE));
            break;
        case 0x88: /* NEWTRUE */
        case 0x89: /* NEWFALSE */
            v = pv_new(m, PV_BOOL);
            v->i = op == 0x88;
            push(m, v);
            break;
        case 'J': /* BININT */
            v = pv_new(m, PV_INT);
            v->i = (int32_t)rd32(take(m, 4));
            push(m, v);
            break;
        case 'K': /* BININT1 */
            v = pv_new(m, PV_INT);
            v->i = *take(m, 1);
            push(m, v);
            break;
        case 'M': /* BININT2 */
            v = pv_new(m, PV_INT);
            v->i = rd16(take(m, 2));
            push(m, v);
            break;
        case 0x8a: { /* LONG1 */
            int n = *take(m, 1);
            const unsigned char *q = take(m, n);
            uint64_t u = 0;

            if (n > 8)
                G_fatal_error(_("Integer too large in checkpoint pickle"));
            for (k = n - 1; k >= 0; k--)
                u = (u << 8) | q[k];
            if (n > 0 && n < 8 && (q[n - 1] & 0x80))
                u |= ~(uint64_t)0 << (8 * n);
            v = pv_new(m, PV_INT);
            v->i = (long)u;
            push(m, v);
            break;
        }
        case 'G': { /* BINFLOAT, big endian */
            const unsigned char *q = take(m, 8);
            uint64_t u = 0;
            double d;

            for (k = 0; k < 8; k++)
                u = (u << 8) | q[k];
            memcpy(&d, &u, 8);
            v = pv_new(m, PV_FLOAT);
            v->f = d;
            push(m, v);
            break;
        }
        case 'X': { /* BINUNICODE */
            uint32_t n = rd32(take(m, 4));

            push(m, make_str(m, take(m, n), n));
            break;
        }
        case 0x8c: { /* SHORT_BINUNICODE */
            unsigned n = *take(m, 1);

            push(m, make_str(m, take(m, n), n));
            break;
        }
        case 0x8d: { /* BINUNICODE8 */
            uint64_t n = rd64(take(m, 8));

            push(m, make_str(m, take(m, n), n));
            break;
        }
        case 'T': { /* BINSTRING */
            uint32_t n = rd32(take(m, 4));

            push(m, make_str(m, take(m, n), n));
            break;
        }
        case 'U': { /* SHORT_BINSTRING */
            unsigned n = *take(m, 1);

            push(m, make_str(m, take(m, n), n));
            break;
        }
        case 'c': /* GLOBAL */
            v = pv_new(m, PV_GLOBAL);
            v->s = take_line(m);
            v->s2 = take_line(m);
            push(m, v);
            break;
        case 0x93: /* STACK_GLOBAL */
            b = pop(m);
            a = pop(m);
            if (a->t != PV_STR || b->t != PV_STR)
                G_fatal_error(_("Invalid STACK_GLOBAL in checkpoint"));
            v = pv_new(m, PV_GLOBAL);
            v->s = G_store(a->s);
            v->s2 = G_store(b->s);
            push(m, v);
            break;
        case 'q': /* BINPUT */
            memo_put(m, *take(m, 1), top(m));
            break;
        case 'r': /* LONG_BINPUT */
            memo_put(m, rd32(take(m, 4)), top(m));
            break;
        case 0x94: /* MEMOIZE */
        {
            long idx = 0;

            while (idx < m->mcap && m->memo[idx])
                idx++;
            memo_put(m, idx, top(m));
            break;
        }
        case 'h': /* BINGET */
            push(m, memo_get(m, *take(m, 1)));
            break;
        case 'j': /* LONG_BINGET */
            push(m, memo_get(m, rd32(take(m, 4))));
            break;
        case 't': /* TUPLE */
            k = pop_mark(m);
            v = make_tuple_from(m, k + 1);
            m->sp = k;
            push(m, v);
            break;
        case 0x85: /* TUPLE1 */
        case 0x86: /* TUPLE2 */
        case 0x87: /* TUPLE3 */
            k = m->sp - (op - 0x84);
            if (k < 0)
                G_fatal_error(_("Pickle stack underflow in checkpoint"));
            push(m, make_tuple_from(m, k));
            break;
        case 'Q': /* BINPERSID */
            push(m, persistent_load(m, pop(m)));
            break;
        case 'R': /* REDUCE */
            a = pop(m);
            b = pop(m);
            push(m, reduce(m, b, a));
            break;
        case 0x81: /* NEWOBJ */
            pop(m);
            a = pop(m);
            v = pv_new(m, PV_OBJ);
            v->cls = a;
            push(m, v);
            break;
        case 'b': /* BUILD */
            a = pop(m);
            v = top(m);
            /* A (dict, slots) pair carries the instance dict first. */
            if (a->t == PV_TUPLE && a->n == 2)
                a = a->items[0];
            if (v->t == PV_OBJ)
                v->state = a;
            break;
        case 's': /* SETITEM */
            b = pop(m);
            a = pop(m);
            v = top(m);
            if (v->t == PV_DICT) {
                pv_append(v, a);
                pv_append(v, b);
            }
            break;
        case 'u': /* SETITEMS */
            k = pop_mark(m);
            if (k < 1 || ((m->sp - k - 1) % 2) != 0)
                G_fatal_error(_("Invalid SETITEMS in checkpoint pickle"));
            v = m->stack[k - 1];
            if (v->t == PV_DICT) {
                int j;

                for (j = k + 1; j < m->sp; j++)
                    pv_append(v, m->stack[j]);
            }
            m->sp = k;
            break;
        case 'a': /* APPEND */
            a = pop(m);
            v = top(m);
            if (v->t == PV_LIST)
                pv_append(v, a);
            break;
        case 'e': /* APPENDS */
            k = pop_mark(m);
            if (k < 1)
                G_fatal_error(_("Invalid APPENDS in checkpoint pickle"));
            v = m->stack[k - 1];
            if (v->t == PV_LIST) {
                int j;

                for (j = k + 1; j < m->sp; j++)
                    pv_append(v, m->stack[j]);
            }
            m->sp = k;
            break;
        default:
            G_fatal_error(_("Unsupported pickle opcode 0x%02x in checkpoint"),
                          op);
        }
    }
}

static int dict_has_tensors(const struct pv *d)
{
    int k;

    for (k = 1; k < d->n; k += 2)
        if (d->items[k]->t == PV_TENSOR)
            return 1;
    return 0;
}

static const struct pv *dict_get(const struct pv *d, const char *key)
{
    int k;

    if (!d || d->t != PV_DICT)
        return NULL;
    for (k = 0; k + 1 < d->n; k += 2)
        if (d->items[k]->t == PV_STR && strcmp(d->items[k]->s, key) == 0)
            return d->items[k + 1];
    return NULL;
}

/* Accept a bare state_dict, or a wrapper dict holding one under "model"
 * or "state_dict" (common for training checkpoints). */
static const struct pv *find_state_dict(const struct pv *root)
{
    int k;

    if (root->t != PV_DICT)
        return NULL;
    if (dict_has_tensors(root))
        return root;
    for (k = 0; k + 1 < root->n; k += 2) {
        const struct pv *key = root->items[k], *val = root->items[k + 1];

        if (key->t == PV_STR && val->t == PV_DICT &&
            (strcmp(key->s, "model") == 0 ||
             strcmp(key->s, "state_dict") == 0) &&
            dict_has_tensors(val))
            return val;
    }
    return NULL;
}

/* The pickled model of a checkpoint that saves a whole torch.nn.Module:
 * under "model", or under "ema" when "model" is None (Ultralytics
 * training checkpoints keep only the EMA weights). */
static const struct pv *find_model(const struct pv *root)
{
    const struct pv *v = dict_get(root, "model");

    if (v && v->t == PV_OBJ)
        return v;
    v = dict_get(root, "ema");
    if (v && v->t == PV_OBJ)
        return v;
    return NULL;
}

/* Named tensors collected from a dict or a module tree. */
struct named {
    int n, cap;
    char **name;
    const struct pv **t;
};

static void named_add(struct named *nt, const char *name, const struct pv *t)
{
    if (nt->n == nt->cap) {
        nt->cap = nt->cap ? 2 * nt->cap : 256;
        nt->name = G_realloc(nt->name, nt->cap * sizeof(char *));
        nt->t = G_realloc(nt->t, nt->cap * sizeof(struct pv *));
    }
    nt->name[nt->n] = G_store(name);
    nt->t[nt->n++] = t;
}

/* Record the plain attributes of a module: numbers, booleans, strings
 * and tuples or lists of integers. */
static void module_attrs(struct pth_module *md, const struct pv *st)
{
    int k, j;

    md->attr = G_calloc(st->n / 2 + 1, sizeof(struct pth_attr));
    for (k = 0; k + 1 < st->n; k += 2) {
        const struct pv *key = st->items[k], *v = st->items[k + 1];
        struct pth_attr *a = &md->attr[md->nattr];

        if (key->t != PV_STR || key->s[0] == '_')
            continue;
        if (v->t == PV_INT || v->t == PV_BOOL) {
            a->type = PA_INT;
            a->v[0] = v->i;
        }
        else if (v->t == PV_FLOAT) {
            a->type = PA_FLOAT;
            a->f = v->f;
        }
        else if (v->t == PV_STR) {
            a->type = PA_STR;
            a->s = G_store(v->s);
        }
        else if ((v->t == PV_TUPLE || v->t == PV_LIST) &&
                 v->n <= PTH_MAX_DIMS) {
            for (j = 0; j < v->n; j++)
                if (v->items[j]->t != PV_INT && v->items[j]->t != PV_BOOL)
                    break;
            if (j < v->n)
                continue;
            a->type = PA_INTS;
            a->n = v->n;
            for (j = 0; j < v->n; j++)
                a->v[j] = v->items[j]->i;
        }
        else
            continue;
        a->name = G_store(key->s);
        md->nattr++;
    }
}

/* Walk a pickled torch.nn.Module: its _parameters and _buffers give the
 * state_dict entries, its _modules the children. */
static void walk_module(struct pth_file *pf, const struct pv *obj,
                        const char *path, struct named *nt, int depth)
{
    static const char *const kinds[] = {"_parameters", "_buffers"};
    const struct pv *st = obj->state, *d;
    struct pth_module *md;
    char *name;
    int k, i;

    if (depth > 64)
        G_fatal_error(_("Module tree too deep in checkpoint"));
    if (!st || st->t != PV_DICT)
        return;

    pf->modules =
        G_realloc(pf->modules, (pf->nmodules + 1) * sizeof(struct pth_module));
    md = &pf->modules[pf->nmodules++];
    memset(md, 0, sizeof(*md));
    md->path = G_store(path);
    md->cls = G_store(obj->cls && obj->cls->t == PV_GLOBAL ? obj->cls->s2 : "");
    module_attrs(md, st);

    for (i = 0; i < 2; i++) {
        d = dict_get(st, kinds[i]);
        if (!d || d->t != PV_DICT)
            continue;
        for (k = 0; k + 1 < d->n; k += 2)
            if (d->items[k]->t == PV_STR && d->items[k + 1]->t == PV_TENSOR) {
                G_asprintf(&name, "%s%s%s", path, *path ? "." : "",
                           d->items[k]->s);
                named_add(nt, name, d->items[k + 1]);
                G_free(name);
            }
    }
    d = dict_get(st, "_modules");
    if (!d || d->t != PV_DICT)
        return;
    for (k = 0; k + 1 < d->n; k += 2)
        if (d->items[k]->t == PV_STR && d->items[k + 1]->t == PV_OBJ) {
            G_asprintf(&name, "%s%s%s", path, *path ? "." : "", d->items[k]->s);
            walk_module(pf, d->items[k + 1], name, nt, depth + 1);
            G_free(name);
        }
}

static size_t dtype_size(enum pth_dtype d) { return d == PTH_F32 ? 4 : 2; }

void pth_open(struct pth_file *pf, const char *path)
{
    struct zip_dir zd;
    const struct zip_entry *pkl = NULL;
    const struct pv *sd, *root, *model;
    struct named nt = {0, 0, NULL, NULL};
    struct pvm m;
    unsigned char *buf;
    char *prefix, *name;
    size_t plen;
    int i, k, nt_count;

    memset(pf, 0, sizeof(*pf));
    pf->path = G_store(path);
    pf->fp = fopen(path, "rb");
    if (!pf->fp)
        G_fatal_error(_("Unable to open checkpoint <%s>"), path);

    zip_read_dir(pf->fp, path, &zd);

    /* The archive root folder is named after the saved file, so locate
     * data.pkl by suffix and derive the prefix from it. */
    for (i = 0; i < zd.n; i++) {
        size_t n = strlen(zd.e[i].name);

        if (n >= 8 && strcmp(zd.e[i].name + n - 8, "data.pkl") == 0 &&
            (n == 8 || zd.e[i].name[n - 9] == '/')) {
            pkl = &zd.e[i];
            break;
        }
    }
    if (!pkl)
        G_fatal_error(_("No data.pkl in <%s>: not a PyTorch checkpoint"), path);
    plen = strlen(pkl->name) - 8;
    prefix = G_malloc(plen + 1);
    memcpy(prefix, pkl->name, plen);
    prefix[plen] = '\0';

    buf = G_malloc(pkl->size);
    read_at(pf->fp, zip_data_offset(pf->fp, path, pkl), buf, pkl->size, path);

    memset(&m, 0, sizeof(m));
    m.p = buf;
    m.end = buf + pkl->size;
    root = run_pickle(&m);
    sd = find_state_dict(root);
    if (sd) {
        for (k = 1; k < sd->n; k += 2)
            if (sd->items[k]->t == PV_TENSOR && sd->items[k - 1]->t == PV_STR)
                named_add(&nt, sd->items[k - 1]->s, sd->items[k]);
    }
    else if ((model = find_model(root)))
        walk_module(pf, model, "", &nt, 0);
    else
        G_fatal_error(_("No state_dict or model found in <%s>"), path);

    nt_count = 0;
    for (k = 0; k < nt.n; k++)
        if (nt.t[k]->storage->dtype != PTH_OTHER)
            nt_count++;
    pf->tensors = G_calloc(nt_count ? nt_count : 1, sizeof(struct pth_tensor));

    name = G_malloc(plen + 256);
    for (k = 0; k < nt.n; k++) {
        const struct pv *tv = nt.t[k];
        struct pth_tensor *t;
        const struct zip_entry *ze;
        long last;
        int d;

        if (tv->storage->dtype == PTH_OTHER)
            continue;
        t = &pf->tensors[pf->ntensors++];
        t->name = G_store(nt.name[k]);
        t->ndim = tv->ndim;
        t->storage_offset = tv->i;
        t->dtype = tv->storage->dtype;
        for (d = 0; d < tv->ndim; d++) {
            t->shape[d] = tv->shape[d];
            t->stride[d] = tv->stride[d];
        }

        snprintf(name, plen + 256, "%sdata/%s", prefix, tv->storage->s);
        ze = zip_find(&zd, name);
        if (!ze)
            G_fatal_error(_("Storage <%s> of tensor <%s> missing in <%s>"),
                          name, t->name, path);
        t->data_offset = zip_data_offset(pf->fp, path, ze);
        t->storage_bytes = ze->size;

        /* Validate that every strided element lies inside the storage. */
        last = t->storage_offset;
        for (d = 0; d < t->ndim; d++) {
            if (t->shape[d] <= 0)
                G_fatal_error(_("Empty tensor <%s> in checkpoint"), t->name);
            last += (t->shape[d] - 1) * t->stride[d];
        }
        if (t->storage_offset < 0 ||
            (size_t)(last + 1) * dtype_size(t->dtype) > t->storage_bytes)
            G_fatal_error(_("Tensor <%s> exceeds its storage in <%s>"), t->name,
                          path);
    }
    G_free(name);
    for (k = 0; k < nt.n; k++)
        G_free(nt.name[k]);
    G_free(nt.name);
    G_free(nt.t);

    for (i = 0; i < m.nall; i++) {
        struct pv *v = m.all[i];

        /* PV_STORAGE keys alias PV_STR payloads and are not freed. */
        if (v->t == PV_STR || v->t == PV_GLOBAL)
            G_free(v->s);
        if (v->t == PV_GLOBAL)
            G_free(v->s2);
        G_free(v->items);
    }
    for (i = 0; i < m.nall; i++)
        G_free(m.all[i]);
    G_free(m.all);
    G_free(m.stack);
    G_free(m.memo);
    G_free(buf);
    G_free(prefix);
    for (i = 0; i < zd.n; i++)
        G_free(zd.e[i].name);
    G_free(zd.e);

    G_verbose_message(_("Checkpoint <%s>: %d tensors, %d modules"), path,
                      pf->ntensors, pf->nmodules);
}

const struct pth_module *pth_module(const struct pth_file *pf, const char *path)
{
    int i;

    for (i = 0; i < pf->nmodules; i++)
        if (strcmp(pf->modules[i].path, path) == 0)
            return &pf->modules[i];
    return NULL;
}

const struct pth_attr *pth_attr(const struct pth_module *m, const char *name)
{
    int i;

    if (!m)
        return NULL;
    for (i = 0; i < m->nattr; i++)
        if (strcmp(m->attr[i].name, name) == 0)
            return &m->attr[i];
    return NULL;
}

const struct pth_tensor *pth_find(const struct pth_file *pf, const char *name)
{
    int i;

    for (i = 0; i < pf->ntensors; i++)
        if (strcmp(pf->tensors[i].name, name) == 0)
            return &pf->tensors[i];
    return NULL;
}

const struct pth_tensor *pth_require(const struct pth_file *pf,
                                     const char *name, int ndim,
                                     const long *shape)
{
    const struct pth_tensor *t = pth_find(pf, name);
    int d;

    if (!t)
        G_fatal_error(_("Tensor <%s> not found in checkpoint <%s>"), name,
                      pf->path);
    if (t->ndim != ndim)
        G_fatal_error(_("Tensor <%s> has rank %d, expected %d"), name, t->ndim,
                      ndim);
    for (d = 0; d < ndim; d++)
        if (shape && shape[d] >= 0 && t->shape[d] != shape[d])
            G_fatal_error(_("Tensor <%s> has size %ld along axis %d, "
                            "expected %ld"),
                          name, t->shape[d], d, shape[d]);
    return t;
}

long pth_numel(const struct pth_tensor *t)
{
    long n = 1;
    int d;

    for (d = 0; d < t->ndim; d++)
        n *= t->shape[d];
    return n;
}

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, u;
    float f;

    if (exp == 0) {
        if (man == 0)
            u = sign;
        else {
            /* Subnormal: renormalize into a float32 normal. */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                exp--;
            }
            u = sign | (exp << 23) | ((man & 0x3ff) << 13);
        }
    }
    else if (exp == 31)
        u = sign | 0x7f800000 | (man << 13);
    else
        u = sign | ((exp + 127 - 15) << 23) | (man << 13);
    memcpy(&f, &u, 4);
    return f;
}

static float elem_to_float(const unsigned char *p, enum pth_dtype d)
{
    float f;
    uint32_t u;

    switch (d) {
    case PTH_F32:
        memcpy(&f, p, 4);
        return f;
    case PTH_F16:
        return half_to_float(rd16(p));
    default:
        u = (uint32_t)rd16(p) << 16;
        memcpy(&f, &u, 4);
        return f;
    }
}

float *pth_read_f32(const struct pth_file *pf, const struct pth_tensor *t)
{
    long n = pth_numel(t), expect = 1, i;
    size_t es = dtype_size(t->dtype);
    int d, contiguous = 1;
    float *out = G_malloc(n * sizeof(float));
    unsigned char *raw;

    for (d = t->ndim - 1; d >= 0; d--) {
        if (t->shape[d] != 1 && t->stride[d] != expect)
            contiguous = 0;
        expect *= t->shape[d];
    }

    if (contiguous) {
        raw = t->dtype == PTH_F32 ? (unsigned char *)out : G_malloc(n * es);
        read_at(pf->fp, t->data_offset + (off_t)t->storage_offset * es, raw,
                n * es, pf->path);
        if (t->dtype != PTH_F32) {
            for (i = 0; i < n; i++)
                out[i] = elem_to_float(raw + i * es, t->dtype);
            G_free(raw);
        }
        return out;
    }

    /* Strided view: read the whole storage and gather in C order. */
    raw = G_malloc(t->storage_bytes);
    read_at(pf->fp, t->data_offset, raw, t->storage_bytes, pf->path);
    {
        long idx[PTH_MAX_DIMS] = {0};

        for (i = 0; i < n; i++) {
            long off = t->storage_offset;

            for (d = 0; d < t->ndim; d++)
                off += idx[d] * t->stride[d];
            out[i] = elem_to_float(raw + off * es, t->dtype);
            for (d = t->ndim - 1; d >= 0; d--) {
                if (++idx[d] < t->shape[d])
                    break;
                idx[d] = 0;
            }
        }
    }
    G_free(raw);
    return out;
}

double pth_bytes(const struct pth_file *pf, const char *prefix)
{
    size_t len = strlen(prefix);
    double sum = 0;
    int i;

    for (i = 0; i < pf->ntensors; i++)
        if (strncmp(pf->tensors[i].name, prefix, len) == 0)
            sum += (double)pth_numel(&pf->tensors[i]) *
                   dtype_size(pf->tensors[i].dtype);
    return sum;
}

void pth_close(struct pth_file *pf)
{
    int i;

    if (pf->fp)
        fclose(pf->fp);
    for (i = 0; i < pf->ntensors; i++)
        G_free(pf->tensors[i].name);
    G_free(pf->tensors);
    for (i = 0; i < pf->nmodules; i++) {
        struct pth_module *md = &pf->modules[i];
        int j;

        for (j = 0; j < md->nattr; j++) {
            G_free(md->attr[j].name);
            G_free(md->attr[j].s);
        }
        G_free(md->attr);
        G_free(md->path);
        G_free(md->cls);
    }
    G_free(pf->modules);
    G_free(pf->path);
    memset(pf, 0, sizeof(*pf));
}
