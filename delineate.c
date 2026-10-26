/****************************************************************************
 *
 * MODULE:       i.delineate.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Delineate-Anything's "main" delineation method (Lavreniuk
 *               et al. 2025): overlapping tiles, per-tile composition of
 *               the detected field masks, merging of the fields across
 *               tiles and field identity opening.
 *
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 *****************************************************************************/

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "delineate.h"
#include "progress.h"

#define TS YOLO_IMG

/* Field identities: union-find whose roots are the smallest identity of
 * each set (as Delineate-Anything's IncrementalFastMapper), and the
 * composed area of each identity in tile cells. Field identities are
 * even; identity | 1 marks the same field within the merging edge of its
 * tile. */
struct ids {
    int *parent;
    long *area;
    long cap;
    int next;
};

static void ids_reserve(struct ids *s, long n)
{
    long i, cap = s->cap;

    if (n <= cap)
        return;
    while (cap < n)
        cap = cap ? 2 * cap : 1 << 16;
    if (cap > INT32_MAX)
        G_fatal_error(_("Too many field identities"));
    s->parent = G_realloc(s->parent, cap * sizeof(int));
    s->area = G_realloc(s->area, cap * sizeof(long));
    for (i = s->cap; i < cap; i++) {
        s->parent[i] = (int)i;
        s->area[i] = 0;
    }
    s->cap = cap;
}

static int uf_find(int *parent, int a)
{
    while (parent[a] != a) {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    return a;
}

static void uf_union(int *parent, int a, int b)
{
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a < b)
        parent[b] = a;
    else if (b < a)
        parent[a] = b;
}

/* Open-addressing counters keyed by 64-bit values. */
struct counter {
    uint64_t *key;
    long *cnt;
    long cap, n;
};

#define EMPTY_KEY UINT64_MAX

static void counter_init(struct counter *h, long cap)
{
    long i;

    h->cap = cap;
    h->n = 0;
    h->key = G_malloc(cap * sizeof(uint64_t));
    h->cnt = G_calloc(cap, sizeof(long));
    for (i = 0; i < cap; i++)
        h->key[i] = EMPTY_KEY;
}

static long counter_slot(const struct counter *h, uint64_t k)
{
    uint64_t x = k * 0x9E3779B97F4A7C15ULL;
    long i = (long)(x >> 20) & (h->cap - 1);

    while (h->key[i] != EMPTY_KEY && h->key[i] != k)
        i = (i + 1) & (h->cap - 1);
    return i;
}

static void counter_add(struct counter *h, uint64_t k, long v)
{
    long i;

    if (2 * (h->n + 1) > h->cap) {
        struct counter g;
        long j;

        counter_init(&g, 2 * h->cap);
        for (j = 0; j < h->cap; j++)
            if (h->key[j] != EMPTY_KEY) {
                long s = counter_slot(&g, h->key[j]);

                g.key[s] = h->key[j];
                g.cnt[s] = h->cnt[j];
                g.n++;
            }
        G_free(h->key);
        G_free(h->cnt);
        *h = g;
    }
    i = counter_slot(h, k);
    if (h->key[i] == EMPTY_KEY) {
        h->key[i] = k;
        h->n++;
    }
    h->cnt[i] += v;
}

static long counter_get(const struct counter *h, uint64_t k)
{
    long i = counter_slot(h, k);

    return h->key[i] == k ? h->cnt[i] : 0;
}

static void counter_free(struct counter *h)
{
    G_free(h->key);
    G_free(h->cnt);
}

/* A detected field cropped to its window. */
struct field {
    int x0, y0, w, h;
    unsigned char *m;
    long area;
    double rel_area;
};

/* Moore neighbourhood, clockwise in image coordinates from west. */
static const int MX[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
static const int MY[8] = {0, -1, -1, -1, 0, 1, 1, 1};

static int moore_dir(int dx, int dy)
{
    int d;

    for (d = 0; d < 8; d++)
        if (MX[d] == dx && MY[d] == dy)
            return d;
    return 0;
}

/* Outer border of component c of lab (8-connected), traced from its
 * first cell in raster order (x0, y0) with Moore neighbour tracing and
 * Jacob's stopping criterion. Returns the number of border cells written
 * to path (x, y pairs, grown as needed). */
static int trace_border(const int *lab, int w, int h, int c, int x0, int y0,
                        int **path, int *cap)
{
    int n = 0, cx = x0, cy = y0, back = 0, fpx = -1, fpy = -1, fcx = -1,
        fcy = -1;

#define PUSH(x, y)                                                             \
    do {                                                                       \
        if (n == *cap) {                                                       \
            *cap = *cap ? 2 * *cap : 256;                                      \
            *path = G_realloc(*path, 2 * *cap * sizeof(int));                  \
        }                                                                      \
        (*path)[2 * n] = (x);                                                  \
        (*path)[2 * n + 1] = (y);                                              \
        n++;                                                                   \
    } while (0)

    PUSH(x0, y0);
    for (;;) {
        int k, d = -1, px, py, pd;

        for (k = 1; k <= 8; k++) {
            int dd = (back + k) % 8, nx = cx + MX[dd], ny = cy + MY[dd];

            if (nx >= 0 && nx < w && ny >= 0 && ny < h &&
                lab[ny * w + nx] == c) {
                d = dd;
                break;
            }
        }
        if (d < 0)
            break; /* A single cell. */
        px = cx;
        py = cy;
        cx += MX[d];
        cy += MY[d];
        pd = (d + 7) % 8;
        back = moore_dir(px + MX[pd] - cx, py + MY[pd] - cy);
        if (fpx < 0) {
            fpx = px;
            fpy = py;
            fcx = cx;
            fcy = cy;
        }
        else if (px == fpx && py == fpy && cx == fcx && cy == fcy)
            break;
        PUSH(cx, cy);
    }
#undef PUSH
    /* The closing move back to the start was pushed last. */
    if (n > 1 && (*path)[2 * (n - 1)] == x0 && (*path)[2 * (n - 1) + 1] == y0)
        n--;
    return n;
}

/* Delineate-Anything keeps, of each detected mask, the outer contour of
 * largest area (cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE),
 * cv2.contourArea, numpy argmax) filled with cv2.fillPoly(LINE_4). The
 * filled contour is the component with its holes plus, for every diagonal
 * step of the contour, the cell right of the step's left end, which the
 * 4-connected line drawing adds. Equal areas go to the contour found last
 * in raster order, the first one in OpenCV's output. */
static void biggest_component(unsigned char *m, int w, int h)
{
    size_t n = (size_t)w * h, i;
    int *lab = G_malloc(n * sizeof(int)), *stack = G_malloc(n * sizeof(int));
    int *path = NULL, *best_path = NULL, cap = 0, best_cap = 0, nbest = 0;
    int nl = 0, best = -1;
    double best_area = -1;

    for (i = 0; i < n; i++)
        lab[i] = -1;
    for (i = 0; i < n; i++) {
        size_t sp = 0;
        int np, k;
        double a = 0;

        if (!m[i] || lab[i] >= 0)
            continue;
        lab[i] = nl;
        stack[sp++] = (int)i;
        while (sp) {
            int c = stack[--sp], y = c / w, x = c % w, dy, dx;

            for (dy = -1; dy <= 1; dy++)
                for (dx = -1; dx <= 1; dx++) {
                    int yy = y + dy, xx = x + dx, q;

                    if (yy < 0 || yy >= h || xx < 0 || xx >= w)
                        continue;
                    q = yy * w + xx;
                    if (m[q] && lab[q] < 0) {
                        lab[q] = nl;
                        stack[sp++] = q;
                    }
                }
        }
        np = trace_border(lab, w, h, nl, (int)(i % w), (int)(i / w), &path,
                          &cap);
        for (k = 0; k < np; k++) {
            int j = (k + 1) % np;

            a += (double)path[2 * k] * path[2 * j + 1] -
                 (double)path[2 * j] * path[2 * k + 1];
        }
        a = fabs(a) / 2;
        if (a >= best_area) {
            int *t = best_path, tc = best_cap;

            best_area = a;
            best = nl;
            nbest = np;
            best_path = path;
            best_cap = cap;
            path = t;
            cap = tc;
        }
        nl++;
    }

    /* Outside: cells 4-connected to the window border without crossing
     * the component. */
    {
        size_t sp = 0;
        unsigned char *out = G_calloc(n, 1);
        int x, y, k;

#define SEED(q)                                                                \
    do {                                                                       \
        if (lab[q] != best && !out[q]) {                                       \
            out[q] = 1;                                                        \
            stack[sp++] = (q);                                                 \
        }                                                                      \
    } while (0)
        for (x = 0; x < w; x++) {
            SEED(x);
            SEED((h - 1) * w + x);
        }
        for (y = 0; y < h; y++) {
            SEED(y * w);
            SEED(y * w + w - 1);
        }
        while (sp) {
            int c = stack[--sp], cy = c / w, cx = c % w;

            if (cx > 0)
                SEED(c - 1);
            if (cx < w - 1)
                SEED(c + 1);
            if (cy > 0)
                SEED(c - w);
            if (cy < h - 1)
                SEED(c + w);
        }
#undef SEED
        for (i = 0; i < n; i++)
            m[i] = best >= 0 && !out[i];
        for (k = 0; best >= 0 && k < nbest; k++) {
            int j = (k + 1) % nbest;
            int x0 = best_path[2 * k], y0 = best_path[2 * k + 1];
            int x1 = best_path[2 * j], y1 = best_path[2 * j + 1];

            if (abs(x1 - x0) == 1 && abs(y1 - y0) == 1) {
                if (x0 < x1)
                    m[y0 * w + x0 + 1] = 1;
                else
                    m[y1 * w + x1 + 1] = 1;
            }
        }
        G_free(out);
    }
    G_free(lab);
    G_free(stack);
    G_free(path);
    G_free(best_path);
}

struct by_area {
    int j;
    long area;
};

/* Ascending area, ties by index: reversed, numpy's argsort()[::-1]. */
static int cmp_area(const void *a, const void *b)
{
    const struct by_area *x = a, *y = b;

    if (x->area != y->area)
        return x->area < y->area ? -1 : 1;
    return x->j - y->j;
}

static int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;

    return x < y ? -1 : x > y;
}

struct region {
    int W, H; /* Output grid. */
    int *inst;
    float *wts;
};

/* Merge decisions between the identities already in the region window
 * (cur) and those of the new tile (nw), as find_edge_mapping. */
static void edge_mapping(const struct region *rg, const int *tile, int bx,
                         int by, int px0, int px1, int py0, int py1,
                         struct ids *s, const struct del_params *p)
{
    struct counter pairs, ucur, unew;
    long i;
    int x, y;

    counter_init(&pairs, 4096);
    counter_init(&ucur, 1024);
    counter_init(&unew, 1024);
    for (y = py0; y < py1; y++)
        for (x = px0; x < px1; x++) {
            uint32_t cv = (uint32_t)rg->inst[(size_t)y * rg->W + x];
            uint32_t nv = (uint32_t)tile[(y - by) * TS + x - bx];

            counter_add(&pairs, ((uint64_t)cv << 32) | nv, 1);
            counter_add(&ucur, cv, 1);
            counter_add(&unew, nv, 1);
        }

    for (i = 0; i < pairs.cap; i++) {
        uint64_t k = pairs.key[i];
        uint32_t kc, kn;
        uint64_t kc0, kc1, kn0, kn1;
        long inter, ac, an, lc, ln;
        double iou, rc, rn;
        int case0, case1;

        if (k == EMPTY_KEY)
            continue;
        kc = (uint32_t)(k >> 32);
        kn = (uint32_t)(k & 0xffffffffu);
        if (kc < 2 || kn < 2)
            continue;
        kc0 = kc & ~1u;
        kc1 = kc0 | 1;
        kn0 = kn & ~1u;
        kn1 = kn0 | 1;
        inter = counter_get(&pairs, (kc0 << 32) | kn0) +
                counter_get(&pairs, (kc0 << 32) | kn1) +
                counter_get(&pairs, (kc1 << 32) | kn0) +
                counter_get(&pairs, (kc1 << 32) | kn1);
        ac = s->area[kc];
        an = s->area[kn];
        iou = inter / (double)(ac + an - inter > 1 ? ac + an - inter : 1);
        case0 = (((kc & 1) || (kn & 1)) &&
                 (iou > p->merge_edge_iou || inter > p->merge_edge_pixels)) ||
                iou > p->merge_iou;
        lc = counter_get(&ucur, kc0) + counter_get(&ucur, kc1);
        ln = counter_get(&unew, kn0) + counter_get(&unew, kn1);
        rc = (double)inter / lc;
        rn = (double)inter / ln;
        case1 = (rc > p->merge_relative_area_threshold &&
                 rn > p->merge_relative_area_threshold) ||
                (inter > p->merge_asymetric_pixel_area_threshold &&
                 rc > p->merge_asymetric_relative_area_threshold) ||
                (inter > p->merge_asymetric_pixel_area_threshold &&
                 rn > p->merge_asymetric_relative_area_threshold);
        G_debug(4,
                "Edge %u %u: inter=%ld ac=%ld an=%ld lc=%ld ln=%ld iou=%.4f "
                "c0=%d c1=%d",
                kc, kn, inter, ac, an, lc, ln, iou, case0, case1);
        if (case0 && case1)
            uf_union(s->parent, (int)kn, (int)kc);
    }
    counter_free(&pairs);
    counter_free(&ucur);
    counter_free(&unew);
}

/* Compose the fields of one tile and merge them into the region, as
 * PostprocWorker.process_fields. clip and filter are the tile masks;
 * (bx, by) is the tile's top-left cell in the output grid. */
static void process_tile(struct region *rg, struct ids *s,
                         const struct yolo_dets *d, const unsigned char *clip,
                         const unsigned char *filter, int bx, int by,
                         const struct del_params *p, int *inst, float *wts)
{
    const int k = d->n, base = s->next;
    const long min_area = p->pixel_area_threshold;
    struct field *f;
    struct by_area *order;
    long *id_area, *cnt;
    int *write_id, *touched, ntouched, j, o, x, y, e;
    int px0, px1, py0, py1;

    if (k == 0)
        return;
    s->next += 2 * k;
    ids_reserve(s, (long)s->next + 2);

    /* Clean the fields: largest component, holes filled, clipped. */
    f = G_calloc(k, sizeof(struct field));
#pragma omp parallel for private(x, y) schedule(dynamic)
    for (j = 0; j < k; j++) {
        const struct yolo_det *dt = &d->v[j];
        long area = 0, valid = 0;

        f[j].x0 = dt->wx0;
        f[j].y0 = dt->wy0;
        f[j].w = dt->ww;
        f[j].h = dt->wh;
        f[j].m = G_malloc((size_t)dt->ww * dt->wh);
        memcpy(f[j].m, dt->mask, (size_t)dt->ww * dt->wh);
        biggest_component(f[j].m, f[j].w, f[j].h);
        for (y = 0; y < f[j].h; y++)
            for (x = 0; x < f[j].w; x++) {
                size_t q = (size_t)(f[j].y0 + y) * TS + f[j].x0 + x;
                unsigned char *v = &f[j].m[y * f[j].w + x];

                *v = *v && clip[q];
                area += *v;
                valid += *v && filter[q];
            }
        f[j].area = area;
        f[j].rel_area = area ? (double)valid / area : 0.0;
    }

    order = G_malloc(k * sizeof(struct by_area));
    for (j = 0; j < k; j++) {
        order[j].j = j;
        order[j].area = f[j].area;
    }
    qsort(order, k, sizeof(struct by_area), cmp_area);

    id_area = G_malloc(k * sizeof(long));
    write_id = G_malloc(k * sizeof(int));
    cnt = G_calloc(k, sizeof(long));
    touched = G_malloc(k * sizeof(int));
    for (j = 0; j < k; j++) {
        id_area[j] = f[j].area;
        write_id[j] = base + 2 * j;
    }
    memset(inst, 0, sizeof(int) * TS * TS);
    memset(wts, 0, sizeof(float) * TS * TS);

#define FIELD_CELLS(fj, body)                                                  \
    for (y = 0; y < (fj)->h; y++)                                              \
        for (x = 0; x < (fj)->w; x++)                                          \
            if ((fj)->m[y * (fj)->w + x]) {                                    \
                size_t q = (size_t)((fj)->y0 + y) * TS + (fj)->x0 + x;         \
                body;                                                          \
            }

    /* Compose, largest first: a field mostly covering an already placed
     * one joins it, otherwise it is placed over the others. */
    for (o = k - 1; o >= 0; o--) {
        int jj = order[o].j, id = base + 2 * jj;
        struct field *fj = &f[jj];
        long area = fj->area;

        if (area < min_area || fj->rel_area < p->remaining_area_threshold) {
            write_id[jj] = 0;
            continue;
        }
        ntouched = 0;
        FIELD_CELLS(fj, {
            int v = inst[q];

            if (v) {
                int t = (v - base) / 2;

                if (!cnt[t]++)
                    touched[ntouched++] = t;
            }
        });
        qsort(touched, ntouched, sizeof(int), cmp_int);
        for (e = 0; e < ntouched; e++) {
            int t = touched[e];
            long inter = cnt[t], ak = id_area[t], u;
            double iou;

            if (ak < min_area)
                continue;
            u = area + ak - inter;
            iou = inter / (double)(u > 1 ? u : 1);
            if (iou > p->compose_merge_iou) {
                int key = base + 2 * t;

                id_area[t] += area - inter;
                id_area[jj] = 0;
                FIELD_CELLS(fj, inst[q] = key);
                write_id[jj] = key;
                break;
            }
        }
        if (id_area[jj] >= min_area) {
            FIELD_CELLS(fj, inst[q] = id);
            for (e = 0; e < ntouched; e++)
                id_area[touched[e]] -= cnt[touched[e]];
        }
        for (e = 0; e < ntouched; e++)
            cnt[touched[e]] = 0;
    }

    /* Paint the composed fields, weighted by 1 / area so that smaller
     * fields win where tiles overlap. */
    memset(inst, 0, sizeof(int) * TS * TS);
    for (o = k - 1; o >= 0; o--) {
        int jj = order[o].j, id = write_id[jj], t;
        float wv;
        long area;

        if (id == 0)
            continue;
        t = (id - base) / 2;
        area = id_area[t];
        if (area < min_area)
            continue;
        wv = (float)(1.0 / area);
        FIELD_CELLS(&f[jj], {
            inst[q] = id;
            wts[q] = wv;
        });
        s->area[id] = area;
        s->area[id | 1] = area;
        uf_union(s->parent, id | 1, id);
    }
#undef FIELD_CELLS

    for (j = 0; j < k; j++)
        G_debug(4, "Tile %d,%d: field %d, area %ld, written as %d", bx, by,
                base + 2 * j, f[j].area, write_id[j]);

    /* The merging edge of the tile. */
    e = p->merging_edge_width;
    for (y = 0; y < TS; y++)
        for (x = 0; x < TS; x++)
            if (y < e || y >= TS - e || x < e || x >= TS - e)
                inst[y * TS + x] |= 1;

    px0 = bx > 0 ? bx : 0;
    px1 = bx + TS < rg->W ? bx + TS : rg->W;
    py0 = by > 0 ? by : 0;
    py1 = by + TS < rg->H ? by + TS : rg->H;
    if (px0 < px1 && py0 < py1) {
        edge_mapping(rg, inst, bx, by, px0, px1, py0, py1, s, p);
        for (y = py0; y < py1; y++)
            for (x = px0; x < px1; x++) {
                size_t r = (size_t)y * rg->W + x;
                int q = (y - by) * TS + x - bx;

                if (wts[q] > rg->wts[r]) {
                    rg->inst[r] = inst[q];
                    rg->wts[r] = wts[q];
                }
            }
    }

    for (j = 0; j < k; j++)
        G_free(f[j].m);
    G_free(f);
    G_free(order);
    G_free(id_area);
    G_free(write_id);
    G_free(cnt);
    G_free(touched);
}

/* Python floor division. */
static int floordiv(int a, int b)
{
    int q = a / b;

    return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q;
}

struct plan {
    int t, step, n, begin;
};

static void make_plan(const struct del_params *p, struct plan *pl)
{
    pl->t = TS / p->scale;
    pl->step = (int)(pl->t * p->tile_step);
    if (pl->step < 1)
        pl->step = 1;
    pl->n = pl->t / pl->step + (pl->t % pl->step != 0);
    pl->begin = floordiv(-(pl->t - pl->step), pl->step) * pl->step;
}

/* Tiles in the order of ExecutionPlanner and DataLoaderCached: one phase
 * of non-overlapping tiles after the other, rows of each phase from top
 * to bottom. Returns the count; fills xy when not NULL. */
static long list_tiles(const struct del_image *img, const struct plan *pl,
                       int *xy)
{
    long n = 0;
    int i, j, x, y, big = pl->n * pl->step;

    for (i = 0; i < pl->n; i++)
        for (j = 0; j < pl->n; j++)
            for (y = pl->begin + j * pl->step; y < img->rows; y += big)
                for (x = pl->begin + i * pl->step; x < img->cols; x += big) {
                    if (xy) {
                        xy[2 * n] = x;
                        xy[2 * n + 1] = y;
                    }
                    n++;
                }
    return n;
}

long delineate_count_tiles(const struct del_image *img,
                           const struct del_params *p)
{
    struct plan pl;

    make_plan(p, &pl);
    return list_tiles(img, &pl, NULL);
}

/* Field identity opening on the whole raster (PostprocHandler's
 * __id_opening with a cross-shaped neighbourhood): cells on the border
 * between identities are cleared, gaps are filled twice from the largest
 * neighbouring identity, and cells next to a remaining gap are cleared.
 * Neighbours outside the raster are ignored. */
static void id_opening(int *d, int *t, int W, int H)
{
    int x, y, pass;

#define NB_LOOP(stmt)                                                          \
    do {                                                                       \
        stmt(0, 0);                                                            \
        if (x > 0)                                                             \
            stmt(0, -1);                                                       \
        if (x < W - 1)                                                         \
            stmt(0, 1);                                                        \
        if (y > 0)                                                             \
            stmt(-1, 0);                                                       \
        if (y < H - 1)                                                         \
            stmt(1, 0);                                                        \
    } while (0)

#pragma omp parallel for private(x)
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++) {
            size_t i = (size_t)y * W + x;
            int mx = d[i], mn = d[i];

#define MINMAX(dy, dx)                                                         \
    do {                                                                       \
        int v_ = d[i + (dy) * (long)W + (dx)];                                 \
        if (v_ > mx)                                                           \
            mx = v_;                                                           \
        if (v_ < mn)                                                           \
            mn = v_;                                                           \
    } while (0)
            NB_LOOP(MINMAX);
#undef MINMAX
            t[i] = mx != mn ? 0 : d[i];
        }

    for (pass = 0; pass < 2; pass++) {
        const int *src = pass == 0 ? t : d;
        int *dst = pass == 0 ? d : t;

#pragma omp parallel for private(x)
        for (y = 0; y < H; y++)
            for (x = 0; x < W; x++) {
                size_t i = (size_t)y * W + x;
                int mx = src[i];

                if (src[i] != 0) {
                    dst[i] = src[i];
                    continue;
                }
#define MAXV(dy, dx)                                                           \
    do {                                                                       \
        int v_ = src[i + (dy) * (long)W + (dx)];                               \
        if (v_ > mx)                                                           \
            mx = v_;                                                           \
    } while (0)
                NB_LOOP(MAXV);
#undef MAXV
                dst[i] = mx;
            }
    }

#pragma omp parallel for private(x)
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++) {
            size_t i = (size_t)y * W + x;
            int z = 0;

#define ZERO(dy, dx)                                                           \
    do {                                                                       \
        if (t[i + (dy) * (long)W + (dx)] == 0)                                 \
            z = 1;                                                             \
    } while (0)
            NB_LOOP(ZERO);
#undef ZERO
            d[i] = z ? 0 : t[i];
        }
#undef NB_LOOP
}

static void dump_dets(FILE *fp, int tx, int ty, const struct yolo_dets *d)
{
    int i;

    for (i = 0; i < d->n; i++) {
        const struct yolo_det *v = &d->v[i];
        long a = 0, q;

        for (q = 0; q < (long)v->ww * v->wh; q++)
            a += v->mask[q];
        fprintf(fp, "%d,%d,%.3f,%.3f,%.3f,%.3f,%.5f,%ld\n", tx, ty, v->x1,
                v->y1, v->x2, v->y2, v->score, a);
    }
}

int *delineate_run(struct yolo_model *m, const struct del_image *img,
                   const struct del_params *p)
{
    struct plan pl;
    struct region rg;
    struct ids s;
    long ntiles, t, done = 0, nproc = 0, ndet = 0;
    int *xy, *slot_xy[2], nbs[2], nb = 0, cur = 0, pending = -1, k, *inst;
    float *planes = NULL, *wts;
    unsigned char *clip[2], *filter[2];
    const size_t plane = (size_t)TS * TS;
    FILE *dump = NULL;
    size_t i, n;

    make_plan(p, &pl);
    ntiles = list_tiles(img, &pl, NULL);
    xy = G_malloc(sizeof(int) * 2 * (ntiles ? ntiles : 1));
    list_tiles(img, &pl, xy);
    G_message(_("Delineating %ld tiles of %d x %d cells (step %d)..."), ntiles,
              pl.t, pl.t, pl.step);

    rg.W = img->cols * p->scale;
    rg.H = img->rows * p->scale;
    n = (size_t)rg.W * rg.H;
    rg.inst = G_calloc(n, sizeof(int));
    rg.wts = G_calloc(n, sizeof(float));
    memset(&s, 0, sizeof(s));
    s.next = 2;
    ids_reserve(&s, 1 << 16);

    for (k = 0; k < 2; k++) {
        clip[k] = G_malloc(plane * p->batch);
        filter[k] = G_malloc(plane * p->batch);
        slot_xy[k] = G_malloc(sizeof(int) * 2 * p->batch);
    }
    inst = G_malloc(sizeof(int) * plane);
    wts = G_malloc(sizeof(float) * plane);
    if (p->dump_dir) {
        char path[GPATH_MAX];

        snprintf(path, sizeof(path), "%s/detections.csv", p->dump_dir);
        dump = fopen(path, "w");
        if (!dump)
            G_fatal_error(_("Unable to write debug dump <%s>"), path);
        fprintf(dump, "tile_x,tile_y,x1,y1,x2,y2,score,mask_cells\n");
    }

    /* Batches alternate between the model's slots: the next batch is
     * prepared and submitted while the device works on the previous one,
     * whose fields are then merged. */
#define FINISH(sl)                                                             \
    do {                                                                       \
        int b_;                                                                \
        yolo_wait(m, (sl));                                                    \
        if (p->dump_dir && nproc == 0)                                         \
            yolo_dump(m, p->dump_dir);                                         \
        for (b_ = 0; b_ < nbs[sl]; b_++) {                                     \
            struct yolo_dets d_ = {0, 0, NULL};                                \
            const int *q_ = slot_xy[sl] + 2 * b_;                              \
            yolo_detect(m, (sl), b_, p->conf, p->nms_iou, p->max_det, &d_);    \
            if (dump)                                                          \
                dump_dets(dump, q_[0], q_[1], &d_);                            \
            ndet += d_.n;                                                      \
            process_tile(&rg, &s, &d_, clip[sl] + plane * b_,                  \
                         filter[sl] + plane * b_, q_[0] * p->scale,            \
                         q_[1] * p->scale, p, inst, wts);                      \
            yolo_dets_free(&d_);                                               \
        }                                                                      \
        yolo_release(m, (sl));                                                 \
        nproc += nbs[sl];                                                      \
        done += nbs[sl];                                                       \
        progress_part(done, ntiles);                                           \
    } while (0)

    progress_begin(PG_INFER, ntiles);
    for (t = 0; t <= ntiles; t++) {
        if (t < ntiles) {
            if (image_tile_valid(img, xy[2 * t], xy[2 * t + 1], pl.t)) {
                if (!planes)
                    planes = yolo_input(m, cur);
                image_tile(img, xy[2 * t], xy[2 * t + 1], pl.t, TS,
                           planes + 3 * plane * nb, clip[cur] + plane * nb,
                           filter[cur] + plane * nb);
                slot_xy[cur][2 * nb] = xy[2 * t];
                slot_xy[cur][2 * nb + 1] = xy[2 * t + 1];
                nb++;
            }
            else {
                done++;
                progress_part(done, ntiles);
            }
        }
        if (nb == 0 || (nb < p->batch && t < ntiles))
            continue;

        yolo_submit(m, cur, nb);
        nbs[cur] = nb;
        nb = 0;
        planes = NULL;
        if (m->nslots == 1 || (p->dump_dir && nproc == 0))
            FINISH(cur);
        else {
            if (pending >= 0)
                FINISH(pending);
            pending = cur;
            cur ^= 1;
        }
    }
    if (pending >= 0)
        FINISH(pending);
#undef FINISH
    progress_end();
    if (dump)
        fclose(dump);
    G_message(_("%ld tiles with data, %ld field masks detected"), nproc, ndet);

    G_free(xy);
    for (k = 0; k < 2; k++) {
        G_free(slot_xy[k]);
        G_free(clip[k]);
        G_free(filter[k]);
    }
    G_free(inst);
    G_free(wts);

    /* Resolve the identities and open them. */
    progress_begin(PG_MERGE, 1);
    G_message(_("Merging fields across tiles..."));
    for (i = 0; i < (size_t)s.cap; i++)
        uf_find(s.parent, (int)i);
#pragma omp parallel for
    for (i = 0; i < n; i++)
        rg.inst[i] = s.parent[rg.inst[i]];
    G_free(s.parent);
    G_free(s.area);
    id_opening(rg.inst, (int *)rg.wts, rg.W, rg.H);
    G_free(rg.wts);

    if (img->bg) {
#pragma omp parallel for
        for (i = 0; i < n; i++)
            if (rg.inst[i] < 2) {
                size_t r = i / rg.W / p->scale, c = i % rg.W / p->scale;

                rg.inst[i] = -img->bg[r * img->cols + c];
            }
    }
    progress_end();
    return rg.inst;
}
