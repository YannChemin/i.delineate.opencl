/****************************************************************************
 *
 * MODULE:       i.delineate.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Ultralytics YOLO11 instance segmentation network, as used
 *               by Delineate-Anything (Lavreniuk et al. 2025), on an
 *               OpenCL device: network construction from the checkpoint's
 *               module tree, inference, and the box decoding, non-maximum
 *               suppression and mask assembly of Ultralytics' predictor.
 *
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 *****************************************************************************/

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <grass/config.h>
#if defined(HAVE_CBLAS_H) && defined(HAVE_LIBBLAS)
#include <cblas.h>
#define HAVE_BLAS 1
#endif

#include <grass/gis.h>
#include <grass/glocale.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "yolo_model.h"

static int omp_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

#define WG YOLO_WG
#define TILE YOLO_TILE
#define TILE_K YOLO_TILE_K

#define EP_SILU 1
#define EP_RESIDUAL 2
#define EP_SHUFFLE2 4

const char *const yolo_profile_names[] = {"cpu", "igpu", "dgpu", "egpu"};

/* Below this host link bandwidth (GB/s) a discrete GPU is treated as an
 * eGPU: Thunderbolt 3/4 carries about 3 GB/s, PCIe 3.0 x16 about 16. */
#define SLOW_LINK_GBS 6.0

static int profile_index(const char *name)
{
    int i;

    for (i = 0; i < 4; i++)
        if (strcmp(name, yolo_profile_names[i]) == 0)
            return i;
    G_fatal_error(_("Unknown profile <%s>"), name);
    return -1;
}

static int contains(const char *hay, const char *needle)
{
    size_t n = strlen(needle);

    for (; *hay; hay++)
        if (G_strncasecmp(hay, needle, n) == 0)
            return 1;
    return 0;
}

const char *yolo_device_opts(struct ocl_backend *ocl, void *data)
{
    struct yolo_setup *st = data;
    const char *why;
    char reason[256];

    if (!st->profile_opt || strcmp(st->profile_opt, "auto") == 0) {
        if (!ocl->is_gpu) {
            st->profile = YP_CPU;
            why = _("not a GPU");
        }
        else if (ocl->unified) {
            st->profile = YP_IGPU;
            why = _("GPU sharing memory with the host");
        }
        else if (contains(ocl->platform_name, "remote") ||
                 contains(ocl->device_name, "remote")) {
            st->profile = YP_EGPU;
            why = _("remote device");
        }
        else if (ocl->link_gbs > 0 && ocl->link_gbs < SLOW_LINK_GBS) {
            st->profile = YP_EGPU;
            why = _("slow host link");
        }
        else {
            st->profile = YP_DGPU;
            why = _("discrete GPU");
        }
        snprintf(reason, sizeof(reason), _("auto: %s"), why);
    }
    else {
        st->profile = profile_index(st->profile_opt);
        snprintf(reason, sizeof(reason), _("forced"));
        if (st->profile == YP_CPU && ocl->is_gpu)
            G_warning(_("Profile cpu on GPU <%s>: the convolutions run on "
                        "the host"),
                      ocl->device_name);
        if (st->profile == YP_IGPU && !ocl->unified)
            G_fatal_error(_("Profile igpu needs a device sharing memory with "
                            "the host; <%s> does not"),
                          ocl->device_name);
    }

    if (!st->precision_opt || strcmp(st->precision_opt, "auto") == 0)
        /* Delineate-Anything runs fp16 on GPUs, fp32 on CPUs. */
        st->half = st->profile != YP_CPU;
    else
        st->half = strcmp(st->precision_opt, "fp16") == 0;
#ifdef HAVE_BLAS
    if (st->half && st->profile == YP_CPU)
        G_fatal_error(_("precision=fp16 is not available with profile=cpu "
                        "(host BLAS works in fp32)"));
#endif

    G_verbose_message(
        _("Device profile %s (%s): %s, %u compute units, %s memory, link %s, "
          "%s activations%s"),
        yolo_profile_names[st->profile], reason, ocl->is_gpu ? "GPU" : "CPU",
        ocl->compute_units, ocl->unified ? _("unified") : _("discrete"),
        ocl->link_gbs > 0 ? _("measured") : _("unknown"),
        st->half ? "fp16" : "fp32",
#ifdef HAVE_BLAS
        st->profile == YP_CPU ? _(", host BLAS convolutions") : ""
#else
        st->profile == YP_CPU ? _(", no host BLAS in this build") : ""
#endif
    );
    if (ocl->link_gbs > 0)
        G_verbose_message(_("Host link: %.1f GB/s"), ocl->link_gbs);
    snprintf(st->opts, sizeof(st->opts), "%s%s", YOLO_BUILD_OPTS,
             st->half ? " -DACT_HALF" : "");
    return st->opts;
}

static int host_convs(const struct yolo_model *m)
{
#ifdef HAVE_BLAS
    return m->setup.profile == YP_CPU;
#else
    (void)m;
    return 0;
#endif
}

/* Checkpoint access. */

static const struct pth_module *need_module(const struct pth_file *pf,
                                            const char *path)
{
    const struct pth_module *md = pth_module(pf, path);

    if (!md)
        G_fatal_error(_("Module <%s> not found in checkpoint <%s>"), path,
                      pf->path);
    return md;
}

static const char *module_class(const struct pth_file *pf, const char *path)
{
    const struct pth_module *md = pth_module(pf, path);

    return md ? md->cls : "";
}

/* Integer attribute, or its first value for a tuple such as stride. */
static long attr_int(const struct pth_file *pf, const char *path,
                     const char *name, long dflt)
{
    const struct pth_attr *a = pth_attr(pth_module(pf, path), name);

    if (!a)
        return dflt;
    if (a->type == PA_INT || (a->type == PA_INTS && a->n > 0))
        return a->v[0];
    if (a->type == PA_FLOAT)
        return (long)a->f;
    G_fatal_error(_("Attribute <%s> of module <%s> is not a number"), name,
                  path);
    return dflt;
}

static double attr_double(const struct pth_file *pf, const char *path,
                          const char *name, double dflt)
{
    const struct pth_attr *a = pth_attr(pth_module(pf, path), name);

    if (!a)
        return dflt;
    if (a->type == PA_FLOAT)
        return a->f;
    if (a->type == PA_INT)
        return (double)a->v[0];
    G_fatal_error(_("Attribute <%s> of module <%s> is not a number"), name,
                  path);
    return dflt;
}

static float *read_tensor(const struct pth_file *pf, const char *name, int ndim,
                          const long *shape)
{
    const struct pth_tensor *t = pth_require(pf, name, ndim, shape);
    return pth_read_f32(pf, t);
}

/* Weight upload. */

/* Upload weights w (rows x cin * k * k, row-major as torch stores them)
 * transposed and zero-padded to the (kpad, mpad) layout of conv_gemm,
 * with bias b (rows values). */
static void upload_gemm(struct yolo_model *m, struct yolo_conv *c, int rows,
                        const float *w, const float *b, const char *what)
{
    int ktot = c->cin * c->k * c->k, kpad, r, k;
    float *wt, *bp;

    if (host_convs(m)) {
        c->mpad = rows;
        c->hw = G_malloc(sizeof(float) * rows * ktot);
        c->hb = G_malloc(sizeof(float) * rows);
        memcpy(c->hw, w, sizeof(float) * rows * ktot);
        memcpy(c->hb, b, sizeof(float) * rows);
        return;
    }
    kpad = (ktot + TILE_K - 1) / TILE_K * TILE_K;
    c->mpad = (rows + TILE - 1) / TILE * TILE;
    wt = G_calloc((size_t)kpad * c->mpad, sizeof(float));
    bp = G_calloc(c->mpad, sizeof(float));
    for (r = 0; r < rows; r++) {
        for (k = 0; k < ktot; k++)
            wt[(size_t)k * c->mpad + r] = w[(size_t)r * ktot + k];
        bp[r] = b[r];
    }
    c->w = ocl_upload(m->ocl, wt, sizeof(float) * kpad * c->mpad, what);
    c->b = ocl_upload(m->ocl, bp, sizeof(float) * c->mpad, what);
    G_free(wt);
    G_free(bp);
}

/* An Ultralytics Conv or DWConv (convolution, batch normalization,
 * activation) at prefix, batch normalization folded into the weights. A
 * fused checkpoint (convolution with bias, no batch normalization) is
 * accepted too. */
static void load_conv(struct yolo_model *m, const struct pth_file *pf,
                      const char *prefix, struct yolo_conv *c)
{
    char name[256], cpath[256];
    const char *act;
    long shape[4] = {-1, -1, -1, -1}, bshape[1];
    float *w, *b, *gamma, *beta, *mean, *var;
    double eps;
    int groups, i, j, kk, cout;

    snprintf(cpath, sizeof(cpath), "%s.conv", prefix);
    snprintf(name, sizeof(name), "%s.weight", cpath);
    need_module(pf, cpath);
    w = read_tensor(pf, name, 4, shape);
    {
        const struct pth_tensor *t = pth_find(pf, name);

        for (i = 0; i < 4; i++)
            shape[i] = t->shape[i];
    }
    if (shape[2] != shape[3])
        G_fatal_error(_("Convolution <%s> has a non-square kernel"), prefix);
    if (attr_int(pf, cpath, "dilation", 1) != 1)
        G_fatal_error(_("Convolution <%s> is dilated: not supported"), prefix);
    cout = (int)shape[0];
    groups = (int)attr_int(pf, cpath, "groups", 1);
    c->cout = cout;
    c->cin = (int)shape[1] * groups;
    c->k = (int)shape[2];
    c->s = (int)attr_int(pf, cpath, "stride", 1);
    c->pad = (int)attr_int(pf, cpath, "padding", c->k / 2);
    c->shuffle = 0;
    kk = (int)shape[1] * c->k * c->k;

    bshape[0] = cout;
    b = G_calloc(cout, sizeof(float));
    snprintf(name, sizeof(name), "%s.bias", cpath);
    if (pth_find(pf, name)) {
        float *cb = read_tensor(pf, name, 1, bshape);

        memcpy(b, cb, sizeof(float) * cout);
        G_free(cb);
    }
    snprintf(name, sizeof(name), "%s.bn.weight", prefix);
    if (pth_find(pf, name)) {
        char bpath[256];

        gamma = read_tensor(pf, name, 1, bshape);
        snprintf(name, sizeof(name), "%s.bn.bias", prefix);
        beta = read_tensor(pf, name, 1, bshape);
        snprintf(name, sizeof(name), "%s.bn.running_mean", prefix);
        mean = read_tensor(pf, name, 1, bshape);
        snprintf(name, sizeof(name), "%s.bn.running_var", prefix);
        var = read_tensor(pf, name, 1, bshape);
        snprintf(bpath, sizeof(bpath), "%s.bn", prefix);
        eps = attr_double(pf, bpath, "eps", 1e-5);
        for (i = 0; i < cout; i++) {
            double sc = gamma[i] / sqrt((double)var[i] + eps);

            for (j = 0; j < kk; j++)
                w[(size_t)i * kk + j] = (float)(w[(size_t)i * kk + j] * sc);
            b[i] = (float)(beta[i] + (b[i] - mean[i]) * sc);
        }
        G_free(gamma);
        G_free(beta);
        G_free(mean);
        G_free(var);
    }

    snprintf(name, sizeof(name), "%s.act", prefix);
    act = module_class(pf, name);
    if (strcmp(act, "SiLU") == 0)
        c->silu = 1;
    else if (strcmp(act, "Identity") == 0 || !*act)
        c->silu = 0;
    else
        G_fatal_error(_("Activation <%s> of <%s> is not supported"), act,
                      prefix);

    if (groups == 1) {
        c->dw = 0;
        upload_gemm(m, c, cout, w, b, prefix);
    }
    else if (groups == c->cin && groups == cout && c->k == 3 && c->s == 1 &&
             c->pad == 1) {
        c->dw = 1;
        c->mpad = cout;
        if (host_convs(m)) {
            c->hw = G_malloc(sizeof(float) * cout * 9);
            c->hb = G_malloc(sizeof(float) * cout);
            memcpy(c->hw, w, sizeof(float) * cout * 9);
            memcpy(c->hb, b, sizeof(float) * cout);
        }
        else {
            c->w = ocl_upload(m->ocl, w, sizeof(float) * cout * 9, prefix);
            c->b = ocl_upload(m->ocl, b, sizeof(float) * cout, prefix);
        }
    }
    else
        G_fatal_error(_("Grouped convolution <%s> (%d groups) is not "
                        "supported"),
                      prefix, groups);
    G_free(w);
    G_free(b);
}

/* A plain nn.Conv2d with bias and no activation (the last layer of each
 * head branch). */
static void load_conv2d(struct yolo_model *m, const struct pth_file *pf,
                        const char *prefix, struct yolo_conv *c)
{
    char name[256];
    long shape[4] = {-1, -1, -1, -1}, bshape[1];
    const struct pth_tensor *t;
    float *w, *b;

    snprintf(name, sizeof(name), "%s.weight", prefix);
    w = read_tensor(pf, name, 4, shape);
    t = pth_find(pf, name);
    if (attr_int(pf, prefix, "groups", 1) != 1 || t->shape[2] != t->shape[3])
        G_fatal_error(_("Convolution <%s> is not supported"), prefix);
    c->cout = (int)t->shape[0];
    c->cin = (int)t->shape[1];
    c->k = (int)t->shape[2];
    c->s = (int)attr_int(pf, prefix, "stride", 1);
    c->pad = (int)attr_int(pf, prefix, "padding", 0);
    c->dw = c->silu = c->shuffle = 0;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    bshape[0] = c->cout;
    if (pth_find(pf, name))
        b = read_tensor(pf, name, 1, bshape);
    else
        b = G_calloc(c->cout, sizeof(float));
    upload_gemm(m, c, c->cout, w, b, prefix);
    G_free(w);
    G_free(b);
}

/* The 2 x 2, stride 2 nn.ConvTranspose2d of the mask prototype network,
 * as a 1 x 1 convolution to 4 * cout rows written through a 2x pixel
 * shuffle: row co * 4 + dy * 2 + dx holds weight[ci][co][dy][dx]. */
static void load_convt(struct yolo_model *m, const struct pth_file *pf,
                       const char *prefix, struct yolo_conv *c)
{
    char name[256];
    long shape[4] = {-1, -1, 2, 2}, bshape[1];
    const struct pth_tensor *t;
    float *w, *b, *wr, *br;
    int ci, co, s;

    snprintf(name, sizeof(name), "%s.weight", prefix);
    w = read_tensor(pf, name, 4, shape);
    t = pth_find(pf, name);
    if (attr_int(pf, prefix, "stride", 1) != 2 ||
        attr_int(pf, prefix, "padding", 0) != 0 ||
        attr_int(pf, prefix, "groups", 1) != 1)
        G_fatal_error(_("Transposed convolution <%s> is not supported"),
                      prefix);
    c->cin = (int)t->shape[0];
    c->cout = (int)t->shape[1];
    c->k = 1;
    c->s = 1;
    c->pad = 0;
    c->dw = c->silu = 0;
    c->shuffle = 1;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    bshape[0] = c->cout;
    b = pth_find(pf, name) ? read_tensor(pf, name, 1, bshape)
                           : G_calloc(c->cout, sizeof(float));
    wr = G_malloc(sizeof(float) * 4 * c->cout * c->cin);
    br = G_malloc(sizeof(float) * 4 * c->cout);
    for (co = 0; co < c->cout; co++)
        for (s = 0; s < 4; s++) {
            for (ci = 0; ci < c->cin; ci++)
                wr[((size_t)co * 4 + s) * c->cin + ci] =
                    w[((size_t)ci * c->cout + co) * 4 + s];
            br[co * 4 + s] = b[co];
        }
    upload_gemm(m, c, 4 * c->cout, wr, br, prefix);
    G_free(w);
    G_free(b);
    G_free(wr);
    G_free(br);
}

/* A Sequential of Conv, DWConv, Conv2d and nested Sequentials, flattened
 * into a chain of convolutions. */
static void load_chain(struct yolo_model *m, const struct pth_file *pf,
                       const char *prefix, struct yolo_chain *ch)
{
    char path[256];
    const char *cls;
    int i;

    for (i = 0;; i++) {
        snprintf(path, sizeof(path), "%s.%d", prefix, i);
        cls = module_class(pf, path);
        if (!*cls)
            break;
        if (strcmp(cls, "Sequential") == 0) {
            load_chain(m, pf, path, ch);
            continue;
        }
        ch->v = G_realloc(ch->v, (ch->n + 1) * sizeof(struct yolo_conv));
        memset(&ch->v[ch->n], 0, sizeof(struct yolo_conv));
        if (strcmp(cls, "Conv") == 0 || strcmp(cls, "DWConv") == 0)
            load_conv(m, pf, path, &ch->v[ch->n]);
        else if (strcmp(cls, "Conv2d") == 0)
            load_conv2d(m, pf, path, &ch->v[ch->n]);
        else
            G_fatal_error(_("Module <%s> of class %s is not supported in a "
                            "head branch"),
                          path, cls);
        ch->n++;
    }
    if (ch->n == 0)
        G_fatal_error(_("Empty head branch <%s> in checkpoint"), prefix);
}

static void load_bottleneck(struct yolo_model *m, const struct pth_file *pf,
                            const char *prefix, struct yolo_bottleneck *b)
{
    char path[256];

    snprintf(path, sizeof(path), "%s.cv1", prefix);
    load_conv(m, pf, path, &b->cv1);
    snprintf(path, sizeof(path), "%s.cv2", prefix);
    load_conv(m, pf, path, &b->cv2);
    b->add = (int)attr_int(pf, prefix, "add", 0);
}

static void load_c2f(struct yolo_model *m, const struct pth_file *pf,
                     const char *prefix, struct yolo_layer *L)
{
    char path[256];
    const char *cls;
    int j, k;

    snprintf(path, sizeof(path), "%s.cv1", prefix);
    load_conv(m, pf, path, &L->cv1);
    snprintf(path, sizeof(path), "%s.cv2", prefix);
    load_conv(m, pf, path, &L->cv2);
    L->c = L->cv1.cout / 2;
    for (L->n = 0;; L->n++) {
        snprintf(path, sizeof(path), "%s.m.%d", prefix, L->n);
        if (!pth_module(pf, path))
            break;
    }
    if (L->cv2.cin != (2 + L->n) * L->c)
        G_fatal_error(_("Unexpected channel counts in <%s>"), prefix);
    L->items = G_calloc(L->n ? L->n : 1, sizeof(struct yolo_item));
    for (j = 0; j < L->n; j++) {
        struct yolo_item *it = &L->items[j];

        snprintf(path, sizeof(path), "%s.m.%d", prefix, j);
        cls = module_class(pf, path);
        if (strcmp(cls, "Bottleneck") == 0) {
            it->c3k = 0;
            load_bottleneck(m, pf, path, &it->b);
        }
        else if (strcmp(cls, "C3k") == 0 || strcmp(cls, "C3") == 0) {
            char sub[256];

            it->c3k = 1;
            snprintf(sub, sizeof(sub), "%s.cv1", path);
            load_conv(m, pf, sub, &it->cv1);
            snprintf(sub, sizeof(sub), "%s.cv2", path);
            load_conv(m, pf, sub, &it->cv2);
            snprintf(sub, sizeof(sub), "%s.cv3", path);
            load_conv(m, pf, sub, &it->cv3);
            for (it->nb = 0;; it->nb++) {
                snprintf(sub, sizeof(sub), "%s.m.%d", path, it->nb);
                if (!pth_module(pf, sub))
                    break;
            }
            it->m =
                G_calloc(it->nb ? it->nb : 1, sizeof(struct yolo_bottleneck));
            for (k = 0; k < it->nb; k++) {
                snprintf(sub, sizeof(sub), "%s.m.%d", path, k);
                load_bottleneck(m, pf, sub, &it->m[k]);
            }
        }
        else
            G_fatal_error(_("Module <%s> of class %s is not supported"), path,
                          cls);
    }
}

static void load_c2psa(struct yolo_model *m, const struct pth_file *pf,
                       const char *prefix, struct yolo_layer *L)
{
    char path[256], sub[256];
    int j;

    snprintf(path, sizeof(path), "%s.cv1", prefix);
    load_conv(m, pf, path, &L->cv1);
    snprintf(path, sizeof(path), "%s.cv2", prefix);
    load_conv(m, pf, path, &L->cv2);
    L->c = L->cv1.cout / 2;
    for (L->n = 0;; L->n++) {
        snprintf(path, sizeof(path), "%s.m.%d", prefix, L->n);
        if (!pth_module(pf, path))
            break;
    }
    L->psa = G_calloc(L->n ? L->n : 1, sizeof(struct yolo_psa));
    for (j = 0; j < L->n; j++) {
        struct yolo_psa *p = &L->psa[j];

        snprintf(path, sizeof(path), "%s.m.%d.attn", prefix, j);
        p->heads = (int)attr_int(pf, path, "num_heads", 0);
        p->kd = (int)attr_int(pf, path, "key_dim", 0);
        p->hd = (int)attr_int(pf, path, "head_dim", 0);
        p->scale = (float)attr_double(pf, path, "scale", 0);
        if (p->heads <= 0 || p->kd <= 0 || p->hd <= 0 || p->hd > 64 ||
            p->heads * p->hd != L->c)
            G_fatal_error(_("Unsupported attention <%s>"), path);
        snprintf(sub, sizeof(sub), "%s.qkv", path);
        load_conv(m, pf, sub, &p->qkv);
        snprintf(sub, sizeof(sub), "%s.pe", path);
        load_conv(m, pf, sub, &p->pe);
        snprintf(sub, sizeof(sub), "%s.proj", path);
        load_conv(m, pf, sub, &p->proj);
        if (!p->pe.dw || p->qkv.cout != p->heads * (2 * p->kd + p->hd))
            G_fatal_error(_("Unsupported attention <%s>"), path);
        snprintf(path, sizeof(path), "%s.m.%d.ffn.0", prefix, j);
        load_conv(m, pf, path, &p->ffn0);
        snprintf(path, sizeof(path), "%s.m.%d.ffn.1", prefix, j);
        load_conv(m, pf, path, &p->ffn1);
        snprintf(path, sizeof(path), "%s.m.%d", prefix, j);
        p->add = (int)attr_int(pf, path, "add", 1);
    }
}

static void load_segment(struct yolo_model *m, const struct pth_file *pf,
                         const char *prefix)
{
    struct yolo_head *h = &m->head;
    char path[256];
    int l;

    h->nl = (int)attr_int(pf, prefix, "nl", 0);
    h->nc = (int)attr_int(pf, prefix, "nc", 0);
    h->nm = (int)attr_int(pf, prefix, "nm", 0);
    h->reg_max = (int)attr_int(pf, prefix, "reg_max", 16);
    if (h->nl < 1 || h->nl > 4 || h->nc < 1 || h->nm < 1 || h->reg_max < 1)
        G_fatal_error(_("Unsupported segmentation head <%s>"), prefix);
    if (attr_int(pf, prefix, "end2end", 0))
        G_fatal_error(_("End-to-end heads are not supported"));
    for (l = 0; l < h->nl; l++) {
        snprintf(path, sizeof(path), "%s.cv2.%d", prefix, l);
        load_chain(m, pf, path, &h->box[l]);
        snprintf(path, sizeof(path), "%s.cv3.%d", prefix, l);
        load_chain(m, pf, path, &h->cls[l]);
        snprintf(path, sizeof(path), "%s.cv4.%d", prefix, l);
        load_chain(m, pf, path, &h->mc[l]);
        if (h->box[l].v[h->box[l].n - 1].cout != 4 * h->reg_max ||
            h->cls[l].v[h->cls[l].n - 1].cout != h->nc ||
            h->mc[l].v[h->mc[l].n - 1].cout != h->nm)
            G_fatal_error(_("Unexpected output channels in head level %d"), l);
    }
    snprintf(path, sizeof(path), "%s.proto.cv1", prefix);
    load_conv(m, pf, path, &h->p_cv1);
    snprintf(path, sizeof(path), "%s.proto.upsample", prefix);
    load_convt(m, pf, path, &h->p_up);
    snprintf(path, sizeof(path), "%s.proto.cv2", prefix);
    load_conv(m, pf, path, &h->p_cv2);
    snprintf(path, sizeof(path), "%s.proto.cv3", prefix);
    load_conv(m, pf, path, &h->p_cv3);
    if (h->p_cv3.cout != h->nm)
        G_fatal_error(_("Unexpected prototype channels in <%s>"), prefix);
}

/* Inputs of a layer from its "f" attribute: an index or a list. */
static void layer_inputs(const struct pth_file *pf, const char *path,
                         struct yolo_layer *L)
{
    const struct pth_attr *a = pth_attr(pth_module(pf, path), "f");
    int i;

    if (!a)
        G_fatal_error(_("Layer <%s> has no input list"), path);
    if (a->type == PA_INT) {
        L->nf = 1;
        L->f[0] = (int)a->v[0];
    }
    else if (a->type == PA_INTS && a->n >= 1 && a->n <= 4) {
        L->nf = a->n;
        for (i = 0; i < a->n; i++)
            L->f[i] = (int)a->v[i];
    }
    else
        G_fatal_error(_("Unsupported input list of layer <%s>"), path);
}

/* Execution. Every kernel call is skipped while planning, a dry run that
 * sizes the scratch buffer and allocates the layer outputs. */

static void set_view(cl_kernel k, int *a, const struct yolo_view *v)
{
    int off = (int)v->off, bs = (int)v->bs;

    clSetKernelArg(k, (*a)++, sizeof(cl_mem), &v->buf);
    clSetKernelArg(k, (*a)++, sizeof(int), &off);
    clSetKernelArg(k, (*a)++, sizeof(int), &bs);
}

static double wall(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void prof_add(struct yolo_model *m, const char *name, double sec,
                     double flop, double bytes)
{
    int i;

    for (i = 0; i < m->nprof; i++)
        if (strcmp(m->prof[i].name, name) == 0)
            break;
    if (i == m->nprof) {
        if (m->nprof == 16)
            return;
        m->nprof++;
        m->prof[i].name = name;
    }
    m->prof[i].calls++;
    m->prof[i].sec += sec;
    m->prof[i].flop += flop;
    m->prof[i].bytes += bytes;
}

/* Enqueue a kernel; with profiling, time it alone (clFinish and wall
 * clock, which also works where event timestamps are wrong) and account
 * its floating point operations and bytes moved. */
static void run(struct yolo_model *m, cl_kernel k, int dims,
                const size_t *global, const size_t *local, const char *what,
                double flop, double bytes)
{
    double t0 = 0;

    if (m->ocl->profile) {
        clFinish(m->ocl->queue);
        t0 = wall();
    }
    ocl_check(clEnqueueNDRangeKernel(m->ocl->queue, k, dims, NULL, global,
                                     local, 0, NULL, NULL),
              what);
    if (m->ocl->profile) {
        clFinish(m->ocl->queue);
        prof_add(m, what, wall() - t0, flop, bytes);
    }
}

static size_t round_up(size_t n, size_t q) { return (n + q - 1) / q * q; }

static struct yolo_view slice(struct yolo_view v, int c0, int nc)
{
    v.off += (long)c0 * v.h * v.w;
    v.c = nc;
    return v;
}

/* A scratch tensor of c x h x w per image, valid until the scratch is
 * reset at the next layer. */
static struct yolo_view tmp(struct yolo_model *m, int c, int h, int w)
{
    struct yolo_view v;

    v.buf = m->scratch;
    v.off = m->scratch_top;
    v.bs = (long)c * h * w;
    v.c = c;
    v.h = h;
    v.w = w;
    m->scratch_top += v.bs * m->batch;
    if (m->scratch_top > m->scratch_peak) {
        if (!m->planning)
            G_fatal_error(_("Scratch buffer overflow"));
        m->scratch_peak = m->scratch_top;
    }
    return v;
}

static cl_mem alloc_io(struct yolo_model *m, size_t n, const char *what);

/* A persistent tensor, allocated while planning. */
static struct yolo_view persistent(struct yolo_model *m, struct yolo_view *v,
                                   int c, int h, int w, const char *what)
{
    if (m->planning) {
        v->bs = (long)c * h * w;
        if ((size_t)v->bs * m->batch >= (size_t)1 << 31)
            G_fatal_error(_("Activation <%s> too large for batch=%d"), what,
                          m->batch);
        v->buf = alloc_io(m, (size_t)m->esize * v->bs * m->batch, what);
        v->off = 0;
        v->c = c;
        v->h = h;
        v->w = w;
    }
    return *v;
}

#ifdef HAVE_BLAS
/* cpu profile: the whole forward pass runs on the host, every device
 * buffer it touches mapped once (zero copy on a CPU device) until
 * unmap_all() at the end of the pass. */
static float *map_buf(struct yolo_model *m, cl_mem b)
{
    size_t size;
    cl_int err;
    int i;

    for (i = 0; i < m->nmapped; i++)
        if (m->mapped[i] == b)
            return m->mapped_ptr[i];
    if (m->nmapped == YOLO_MAX_MAPPED)
        G_fatal_error(_("Too many mapped buffers"));
    clGetMemObjectInfo(b, CL_MEM_SIZE, sizeof(size), &size, NULL);
    m->mapped[m->nmapped] = b;
    m->mapped_ptr[m->nmapped] = clEnqueueMapBuffer(
        m->ocl->queue, b, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL,
        NULL, &err);
    ocl_check(err, "clEnqueueMapBuffer");
    return m->mapped_ptr[m->nmapped++];
}

static void unmap_all(struct yolo_model *m)
{
    int i;

    for (i = 0; i < m->nmapped; i++)
        ocl_check(clEnqueueUnmapMemObject(m->ocl->queue, m->mapped[i],
                                          m->mapped_ptr[i], 0, NULL, NULL),
                  "clEnqueueUnmapMemObject");
    m->nmapped = 0;
}

static float *host_ptr(struct yolo_model *m, const struct yolo_view *v)
{
    return map_buf(m, v->buf) + v->off;
}

/* Host versions of the small kernels, parallel over planes. */

static void host_dw(struct yolo_model *m, const struct yolo_conv *c,
                    struct yolo_view in, int grp, int gstride, int goff,
                    struct yolo_view out, const struct yolo_view *res, int nb)
{
    const float *pin = host_ptr(m, &in), *pres = res ? host_ptr(m, res) : NULL;
    float *pout = host_ptr(m, &out);
    const int h = out.h, w = out.w, plane = h * w;
    int job;

#pragma omp parallel for schedule(dynamic)
    for (job = 0; job < nb * c->cout; job++) {
        int b = job / c->cout, ch = job % c->cout, x, y;
        int ci = (ch / grp) * gstride + goff + ch % grp;
        const float *src = pin + (long)b * in.bs + (long)ci * plane;
        const float *k = c->hw + ch * 9;
        float *dst = pout + (long)b * out.bs + (long)ch * plane;
        const float *r =
            pres ? pres + (long)b * res->bs + (long)ch * plane : NULL;

        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                float v = c->hb[ch];
                int dy, dx;

                for (dy = -1; dy <= 1; dy++) {
                    int yy = y + dy;

                    if (yy < 0 || yy >= h)
                        continue;
                    for (dx = -1; dx <= 1; dx++) {
                        int xx = x + dx;

                        if (xx >= 0 && xx < w)
                            v += k[(dy + 1) * 3 + dx + 1] * src[yy * w + xx];
                    }
                }
                if (c->silu)
                    v = v / (1.0f + expf(-v));
                if (r)
                    v += r[y * w + x];
                dst[y * w + x] = v;
            }
    }
}

static void host_maxpool(struct yolo_model *m, struct yolo_view in,
                         struct yolo_view out, int win, int nb)
{
    const float *pin = host_ptr(m, &in);
    float *pout = host_ptr(m, &out);
    const int h = in.h, w = in.w, rad = win / 2;
    int job;

#pragma omp parallel for schedule(dynamic)
    for (job = 0; job < nb * in.c; job++) {
        int b = job / in.c, ch = job % in.c, x, y, d;
        const float *src = pin + (long)b * in.bs + (long)ch * h * w;
        float *dst = pout + (long)b * out.bs + (long)ch * h * w;
        float *row = G_malloc(sizeof(float) * h * w);

        /* Separable: rows then columns. */
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                float v = -INFINITY;

                for (d = -rad; d <= rad; d++)
                    if (x + d >= 0 && x + d < w && src[y * w + x + d] > v)
                        v = src[y * w + x + d];
                row[y * w + x] = v;
            }
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                float v = -INFINITY;

                for (d = -rad; d <= rad; d++)
                    if (y + d >= 0 && y + d < h && row[(y + d) * w + x] > v)
                        v = row[(y + d) * w + x];
                dst[y * w + x] = v;
            }
        G_free(row);
    }
}

static void host_upsample(struct yolo_model *m, struct yolo_view in,
                          struct yolo_view out, int nb)
{
    const float *pin = host_ptr(m, &in);
    float *pout = host_ptr(m, &out);
    const int h = in.h, w = in.w;
    int job;

#pragma omp parallel for
    for (job = 0; job < nb * in.c; job++) {
        int b = job / in.c, ch = job % in.c, x, y;
        const float *src = pin + (long)b * in.bs + (long)ch * h * w;
        float *dst = pout + (long)b * out.bs + (long)ch * 4 * h * w;

        for (y = 0; y < 2 * h; y++)
            for (x = 0; x < 2 * w; x++)
                dst[y * 2 * w + x] = src[(y / 2) * w + x / 2];
    }
}

static void host_copy(struct yolo_model *m, struct yolo_view in,
                      struct yolo_view out, int nb)
{
    const float *pin = host_ptr(m, &in);
    float *pout = host_ptr(m, &out);
    int b;

    for (b = 0; b < nb; b++)
        memcpy(pout + (long)b * out.bs, pin + (long)b * in.bs,
               sizeof(float) * in.c * in.h * in.w);
}

static void host_attention(struct yolo_model *m, const struct yolo_psa *p,
                           struct yolo_view qkv, struct yolo_view att, int nb)
{
    const float *pq = host_ptr(m, &qkv);
    float *pa = host_ptr(m, &att);
    const int n = att.h * att.w, kd = p->kd, hd = p->hd;
    int job;

#pragma omp parallel for schedule(dynamic)
    for (job = 0; job < nb * p->heads; job++) {
        int b = job / p->heads, h = job % p->heads, i, j, d, e;
        const float *q = pq + (long)b * qkv.bs + (long)h * (2 * kd + hd) * n;
        const float *k = q + (long)kd * n, *v = q + (long)2 * kd * n;
        float *o = pa + (long)b * att.bs + (long)h * hd * n;
        float *sc = G_malloc(sizeof(float) * n);

        for (i = 0; i < n; i++) {
            float mx = -INFINITY, sum = 0;

            for (j = 0; j < n; j++) {
                float a = 0;

                for (d = 0; d < kd; d++)
                    a += q[d * n + i] * k[d * n + j];
                sc[j] = a * p->scale;
                if (sc[j] > mx)
                    mx = sc[j];
            }
            for (j = 0; j < n; j++) {
                sc[j] = expf(sc[j] - mx);
                sum += sc[j];
            }
            for (e = 0; e < hd; e++) {
                float a = 0;

                for (j = 0; j < n; j++)
                    a += v[e * n + j] * sc[j];
                o[e * n + i] = a / sum;
            }
        }
        G_free(sc);
    }
}

/* Output columns per block of the host convolution: the patches and
 * products of one block stay in a core's cache. */
#define HOST_BLOCK 256

/* BLAS libraries that run their own thread pool would fight the OpenMP
 * threads below; they are asked for one thread when the call exists. */
extern void openblas_set_num_threads(int) __attribute__((weak));
extern void mkl_set_num_threads(int) __attribute__((weak));
extern void bli_thread_set_num_threads(long) __attribute__((weak));

static void blas_single_thread(void)
{
    if (openblas_set_num_threads)
        openblas_set_num_threads(1);
    if (mkl_set_num_threads)
        mkl_set_num_threads(1);
    if (bli_thread_set_num_threads)
        bli_thread_set_num_threads(1);
}

/* cpu profile: a convolution as BLAS matrix products on the host. The
 * output is split into blocks of columns (and of rows when there are too
 * few columns to keep every thread busy, as in the deepest layers),
 * shared by the OpenMP threads; each thread gathers the patches of its
 * column block (im2col, skipped for 1 x 1 convolutions), multiplies them
 * with a single-threaded sgemm and applies the epilogue of the GEMM
 * kernel. */
static void host_conv(struct yolo_model *m, const struct yolo_conv *c,
                      struct yolo_view in, struct yolo_view out,
                      const struct yolo_view *res, int nb, int hout, int wout)
{
    const int rows = c->shuffle ? 4 * c->cout : c->cout;
    const int kk = c->k * c->k, K = c->cin * kk, N = hout * wout;
    const int direct = c->k == 1 && c->s == 1 && c->pad == 0;
    const int plane_in = in.h * in.w, nblk = (N + HOST_BLOCK - 1) / HOST_BLOCK;
    const float *pin = host_ptr(m, &in);
    const float *pres = res ? host_ptr(m, res) : NULL;
    float *pout = host_ptr(m, &out);
    int nrb, rb, job, njobs;

    /* Row blocks of a multiple of 16 rows, enough for two jobs per
     * thread. */
    nrb = (2 * omp_threads() + nb * nblk - 1) / (nb * nblk);
    if (nrb > rows / 16)
        nrb = rows / 16;
    if (nrb < 1)
        nrb = 1;
    rb = (rows + nrb - 1) / nrb;
    rb = (rb + 15) / 16 * 16;
    nrb = (rows + rb - 1) / rb;
    njobs = nb * nblk * nrb;

#pragma omp parallel
    {
        float *col = G_malloc(sizeof(float) * HOST_BLOCK *
                              ((direct ? 0 : (size_t)K) + rb));
        float *T = col + (direct ? 0 : (size_t)K * HOST_BLOCK);
        int *iy0 = G_malloc(sizeof(int) * 2 * HOST_BLOCK),
            *ix0 = iy0 + HOST_BLOCK;

#pragma omp for schedule(dynamic)
        for (job = 0; job < njobs; job++) {
            int b = job / (nblk * nrb), rest = job % (nblk * nrb);
            int n0 = rest / nrb * HOST_BLOCK, r0 = rest % nrb * rb;
            int nc = N - n0 < HOST_BLOCK ? N - n0 : HOST_BLOCK;
            int nr = rows - r0 < rb ? rows - r0 : rb, r, j;
            const float *X = pin + (long)b * in.bs, *B;
            int ldb;

            if (direct) {
                B = X + n0;
                ldb = N;
            }
            else {
                for (j = 0; j < nc; j++) {
                    int n = n0 + j;

                    iy0[j] = n / wout * c->s - c->pad;
                    ix0[j] = n % wout * c->s - c->pad;
                }
                for (r = 0; r < K; r++) {
                    int ci = r / kk, t = r % kk, ky = t / c->k, kx = t % c->k;
                    const float *src = X + (long)ci * plane_in;
                    float *dst = col + (long)r * nc;

                    for (j = 0; j < nc; j++) {
                        int iy = iy0[j] + ky, ix = ix0[j] + kx;

                        dst[j] = (unsigned)iy < (unsigned)in.h &&
                                         (unsigned)ix < (unsigned)in.w
                                     ? src[iy * in.w + ix]
                                     : 0.0f;
                    }
                }
                B = col;
                ldb = nc;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nr, nc, K,
                        1.0f, c->hw + (long)r0 * K, K, B, ldb, 0.0f, T, nc);
            for (r = 0; r < nr; r++) {
                int rr = r0 + r;
                float bias = c->hb[rr];

                for (j = 0; j < nc; j++) {
                    int n = n0 + j;
                    float v = T[(long)r * nc + j] + bias;
                    long o;

                    if (c->silu)
                        v = v / (1.0f + expf(-v));
                    if (c->shuffle) {
                        int y = n / wout, x = n % wout, sub = rr & 3;

                        o = (long)(rr >> 2) * 4 * N +
                            (long)(2 * y + (sub >> 1)) * 2 * wout + 2 * x +
                            (sub & 1);
                    }
                    else
                        o = (long)rr * N + n;
                    if (pres)
                        v += pres[(long)b * res->bs + o];
                    pout[(long)b * out.bs + o] = v;
                }
            }
        }
        G_free(col);
        G_free(iy0);
    }
}
#endif

static void conv(struct yolo_model *m, const struct yolo_conv *c,
                 struct yolo_view in, struct yolo_view out,
                 const struct yolo_view *res, int nb)
{
    int hout = (in.h + 2 * c->pad - c->k) / c->s + 1;
    int wout = (in.w + 2 * c->pad - c->k) / c->s + 1;
    int flags = (c->silu ? EP_SILU : 0) | (res ? EP_RESIDUAL : 0) |
                (c->shuffle ? EP_SHUFFLE2 : 0);
    struct yolo_view r = res ? *res : out;
    int a = 0;
    double flop, bytes;

    if (in.c != c->cin)
        G_fatal_error(_("Convolution input has %d channels, %d expected"), in.c,
                      c->cin);
    if (out.c != c->cout || (c->shuffle ? out.h != 2 * hout || out.w != 2 * wout
                                        : out.h != hout || out.w != wout))
        G_fatal_error(_("Convolution output shape mismatch"));
    flop = 2.0 * c->cout * (c->dw ? 9 : (double)c->cin * c->k * c->k) * hout *
           wout * (c->shuffle ? 4 : 1) * nb;
    bytes = (double)m->esize * nb *
                ((double)in.c * in.h * in.w +
                 (double)out.c * out.h * out.w * (res ? 2 : 1)) +
            4.0 * (c->dw ? 9 : c->cin * c->k * c->k) * c->cout *
                (c->shuffle ? 4 : 1);
    if (m->planning) {
        m->gflop += 1e-9 * flop / nb;
        return;
    }
#ifdef HAVE_BLAS
    if (host_convs(m)) {
        double t0 = m->ocl->profile ? wall() : 0;

        if (c->dw)
            host_dw(m, c, in, c->cout, c->cout, 0, out, res, nb);
        else
            host_conv(m, c, in, out, res, nb, hout, wout);
        if (m->ocl->profile)
            prof_add(m, c->dw ? "dw_host" : "conv_blas", wall() - t0, flop,
                     bytes);
        return;
    }
#endif

    if (c->dw) {
        cl_kernel k = m->k_dw;
        int one = c->cout, zero = 0;
        size_t g[3] = {round_up(out.w, 8), round_up(out.h, 8),
                       (size_t)c->cout * nb};
        size_t l[3] = {8, 8, 1};

        set_view(k, &a, &in);
        clSetKernelArg(k, a++, sizeof(int), &one);
        clSetKernelArg(k, a++, sizeof(int), &one);
        clSetKernelArg(k, a++, sizeof(int), &zero);
        clSetKernelArg(k, a++, sizeof(cl_mem), &c->w);
        clSetKernelArg(k, a++, sizeof(cl_mem), &c->b);
        set_view(k, &a, &out);
        clSetKernelArg(k, a++, sizeof(int), &c->cout);
        clSetKernelArg(k, a++, sizeof(int), &out.h);
        clSetKernelArg(k, a++, sizeof(int), &out.w);
        clSetKernelArg(k, a++, sizeof(int), &flags);
        set_view(k, &a, &r);
        run(m, k, 3, g, l, "dwconv3x3", flop, bytes);
        return;
    }
    {
        cl_kernel k = m->k_gemm;
        size_t g[3], l[3] = {WG, WG, 1};

        g[0] = round_up((size_t)hout * wout, TILE) / TILE * WG;
        g[1] = (size_t)c->mpad / TILE * WG;
        g[2] = nb;
        set_view(k, &a, &in);
        clSetKernelArg(k, a++, sizeof(int), &c->cin);
        clSetKernelArg(k, a++, sizeof(int), &in.h);
        clSetKernelArg(k, a++, sizeof(int), &in.w);
        clSetKernelArg(k, a++, sizeof(cl_mem), &c->w);
        clSetKernelArg(k, a++, sizeof(cl_mem), &c->b);
        set_view(k, &a, &out);
        {
            int rows = c->shuffle ? 4 * c->cout : c->cout;

            clSetKernelArg(k, a++, sizeof(int), &rows);
        }
        clSetKernelArg(k, a++, sizeof(int), &hout);
        clSetKernelArg(k, a++, sizeof(int), &wout);
        clSetKernelArg(k, a++, sizeof(int), &c->k);
        clSetKernelArg(k, a++, sizeof(int), &c->s);
        clSetKernelArg(k, a++, sizeof(int), &c->pad);
        clSetKernelArg(k, a++, sizeof(int), &flags);
        set_view(k, &a, &r);
        run(m, k, 3, g, l, "conv_gemm", flop, bytes);
    }
}

static int conv_hw(const struct yolo_conv *c, int n)
{
    return (n + 2 * c->pad - c->k) / c->s + 1;
}

/* Output view of a convolution applied to in, in scratch memory. */
static struct yolo_view conv_tmp(struct yolo_model *m,
                                 const struct yolo_conv *c, struct yolo_view in)
{
    return tmp(m, c->cout, conv_hw(c, in.h), conv_hw(c, in.w));
}

static void maxpool(struct yolo_model *m, struct yolo_view in,
                    struct yolo_view out, int win, int nb)
{
    cl_kernel k = m->k_pool;
    size_t g[3] = {round_up(in.w, 8), round_up(in.h, 8), (size_t)in.c * nb};
    size_t l[3] = {8, 8, 1};
    int a = 0;

    if (m->planning)
        return;
#ifdef HAVE_BLAS
    if (host_convs(m)) {
        double t0 = m->ocl->profile ? wall() : 0;

        host_maxpool(m, in, out, win, nb);
        if (m->ocl->profile)
            prof_add(m, "maxpool_host", wall() - t0, 0, 0);
        return;
    }
#endif
    set_view(k, &a, &in);
    set_view(k, &a, &out);
    clSetKernelArg(k, a++, sizeof(int), &in.c);
    clSetKernelArg(k, a++, sizeof(int), &in.h);
    clSetKernelArg(k, a++, sizeof(int), &in.w);
    clSetKernelArg(k, a++, sizeof(int), &win);
    run(m, k, 3, g, l, "maxpool", (double)in.c * in.h * in.w * win * win * nb,
        2.0 * m->esize * in.c * in.h * in.w * nb);
}

static void upsample(struct yolo_model *m, struct yolo_view in,
                     struct yolo_view out, int nb)
{
    cl_kernel k = m->k_up;
    size_t g[3] = {round_up(2 * in.w, 8), round_up(2 * in.h, 8),
                   (size_t)in.c * nb};
    size_t l[3] = {8, 8, 1};
    int a = 0;

    if (m->planning)
        return;
#ifdef HAVE_BLAS
    if (host_convs(m)) {
        double t0 = m->ocl->profile ? wall() : 0;

        host_upsample(m, in, out, nb);
        if (m->ocl->profile)
            prof_add(m, "upsample_host", wall() - t0, 0, 0);
        return;
    }
#endif
    set_view(k, &a, &in);
    set_view(k, &a, &out);
    clSetKernelArg(k, a++, sizeof(int), &in.c);
    clSetKernelArg(k, a++, sizeof(int), &in.h);
    clSetKernelArg(k, a++, sizeof(int), &in.w);
    run(m, k, 3, g, l, "upsample2", 0,
        5.0 * m->esize * in.c * in.h * in.w * nb);
}

static void copy_view(struct yolo_model *m, struct yolo_view in,
                      struct yolo_view out, int nb)
{
    cl_kernel k = m->k_copy;
    int n = in.c * in.h * in.w, a = 0;
    size_t g[2] = {round_up(n, 256), nb}, l[2] = {256, 1};

    if (m->planning)
        return;
#ifdef HAVE_BLAS
    if (host_convs(m)) {
        double t0 = m->ocl->profile ? wall() : 0;

        host_copy(m, in, out, nb);
        if (m->ocl->profile)
            prof_add(m, "copy_host", wall() - t0, 0, 0);
        return;
    }
#endif
    set_view(k, &a, &in);
    set_view(k, &a, &out);
    clSetKernelArg(k, a++, sizeof(int), &n);
    run(m, k, 2, g, l, "copy_planes", 0, 2.0 * m->esize * n * nb);
}

static void bottleneck(struct yolo_model *m, const struct yolo_bottleneck *b,
                       struct yolo_view in, struct yolo_view out, int nb)
{
    long mark = m->scratch_top;
    struct yolo_view h = conv_tmp(m, &b->cv1, in);

    conv(m, &b->cv1, in, h, NULL, nb);
    conv(m, &b->cv2, h, out, b->add ? &in : NULL, nb);
    m->scratch_top = mark;
}

/* C3k: cv3(cat(m(cv1(x)), cv2(x))). */
static void c3k(struct yolo_model *m, const struct yolo_item *it,
                struct yolo_view in, struct yolo_view out, int nb)
{
    long mark = m->scratch_top;
    int ch = it->cv1.cout, k;
    struct yolo_view cat = tmp(m, 2 * ch, in.h, in.w);
    struct yolo_view x = tmp(m, ch, in.h, in.w), y;

    conv(m, &it->cv1, in, it->nb ? x : slice(cat, 0, ch), NULL, nb);
    for (k = 0; k < it->nb; k++) {
        y = k == it->nb - 1 ? slice(cat, 0, ch) : tmp(m, ch, in.h, in.w);
        bottleneck(m, &it->m[k], x, y, nb);
        x = y;
    }
    conv(m, &it->cv2, in, slice(cat, ch, it->cv2.cout), NULL, nb);
    conv(m, &it->cv3, cat, out, NULL, nb);
    m->scratch_top = mark;
}

/* C2f / C3k2: cv2(cat(y0, y1, m0(y1), m1(m0(y1)), ...)), with y0, y1 the
 * two halves of cv1(x). */
static void c2f(struct yolo_model *m, const struct yolo_layer *L,
                struct yolo_view in, struct yolo_view out, int nb)
{
    int c = L->c, j;
    struct yolo_view cat = tmp(m, (2 + L->n) * c, in.h, in.w);

    conv(m, &L->cv1, in, slice(cat, 0, 2 * c), NULL, nb);
    for (j = 0; j < L->n; j++) {
        struct yolo_view a = slice(cat, (1 + j) * c, c);
        struct yolo_view b = slice(cat, (2 + j) * c, c);

        if (L->items[j].c3k)
            c3k(m, &L->items[j], a, b, nb);
        else
            bottleneck(m, &L->items[j].b, a, b, nb);
    }
    conv(m, &L->cv2, cat, out, NULL, nb);
}

static void sppf(struct yolo_model *m, const struct yolo_layer *L,
                 struct yolo_view in, struct yolo_view out, int nb)
{
    int c = L->cv1.cout, j;
    struct yolo_view cat = tmp(m, 4 * c, in.h, in.w);

    conv(m, &L->cv1, in, slice(cat, 0, c), NULL, nb);
    for (j = 0; j < 3; j++)
        maxpool(m, slice(cat, j * c, c), slice(cat, (j + 1) * c, c), L->pool,
                nb);
    conv(m, &L->cv2, cat, out, NULL, nb);
}

/* PSABlock on x, in place: x += proj(attention(x) + pe(v)), then
 * x += ffn(x). */
static void psa(struct yolo_model *m, const struct yolo_psa *p,
                struct yolo_view x, int nb)
{
    long mark = m->scratch_top;
    struct yolo_view qkv = conv_tmp(m, &p->qkv, x);
    struct yolo_view att = tmp(m, x.c, x.h, x.w);
    struct yolo_view hid = conv_tmp(m, &p->ffn0, x);
    int n = x.h * x.w, a = 0;

    conv(m, &p->qkv, x, qkv, NULL, nb);
#ifdef HAVE_BLAS
    if (!m->planning && host_convs(m)) {
        double t0 = m->ocl->profile ? wall() : 0;

        host_attention(m, p, qkv, att, nb);
        host_dw(m, &p->pe, qkv, p->hd, 2 * p->kd + p->hd, 2 * p->kd, att, &att,
                nb);
        if (m->ocl->profile)
            prof_add(m, "attn_host", wall() - t0,
                     2.0 * n * n * p->heads * (p->kd + p->hd) * nb, 0);
    }
    else
#endif
        if (!m->planning) {
        cl_kernel k = m->k_attn;
        size_t g[3] = {round_up(n, 64), p->heads, nb}, l[3] = {64, 1, 1};

        set_view(k, &a, &qkv);
        set_view(k, &a, &att);
        clSetKernelArg(k, a++, sizeof(int), &n);
        clSetKernelArg(k, a++, sizeof(int), &p->kd);
        clSetKernelArg(k, a++, sizeof(int), &p->hd);
        clSetKernelArg(k, a++, sizeof(float), &p->scale);
        run(m, k, 3, g, l, "attention",
            2.0 * n * n * p->heads * (p->kd + p->hd) * nb,
            (double)m->esize * n * (qkv.c + x.c) * nb);

        /* att += pe(v), v read from the qkv channels of each head. */
        k = m->k_dw;
        {
            int gstride = 2 * p->kd + p->hd, goff = 2 * p->kd, flags = 2;
            size_t g2[3] = {round_up(x.w, 8), round_up(x.h, 8),
                            (size_t)x.c * nb};
            size_t l2[3] = {8, 8, 1};

            a = 0;
            set_view(k, &a, &qkv);
            clSetKernelArg(k, a++, sizeof(int), &p->hd);
            clSetKernelArg(k, a++, sizeof(int), &gstride);
            clSetKernelArg(k, a++, sizeof(int), &goff);
            clSetKernelArg(k, a++, sizeof(cl_mem), &p->pe.w);
            clSetKernelArg(k, a++, sizeof(cl_mem), &p->pe.b);
            set_view(k, &a, &att);
            clSetKernelArg(k, a++, sizeof(int), &x.c);
            clSetKernelArg(k, a++, sizeof(int), &x.h);
            clSetKernelArg(k, a++, sizeof(int), &x.w);
            clSetKernelArg(k, a++, sizeof(int), &flags);
            set_view(k, &a, &att);
            run(m, k, 3, g2, l2, "dwconv3x3", 18.0 * x.c * n * nb,
                3.0 * m->esize * x.c * n * nb);
        }
    }
    else
        m->gflop += 4e-9 * (double)n * n * p->heads * (p->kd + p->hd);
    conv(m, &p->proj, att, x, p->add ? &x : NULL, nb);
    conv(m, &p->ffn0, x, hid, NULL, nb);
    conv(m, &p->ffn1, hid, x, p->add ? &x : NULL, nb);
    m->scratch_top = mark;
}

static void c2psa(struct yolo_model *m, const struct yolo_layer *L,
                  struct yolo_view in, struct yolo_view out, int nb)
{
    struct yolo_view t = conv_tmp(m, &L->cv1, in);
    int j;

    conv(m, &L->cv1, in, t, NULL, nb);
    for (j = 0; j < L->n; j++)
        psa(m, &L->psa[j], slice(t, L->c, L->c), nb);
    conv(m, &L->cv2, t, out, NULL, nb);
}

static void chain(struct yolo_model *m, const struct yolo_chain *ch,
                  struct yolo_view in, struct yolo_view out, int nb)
{
    struct yolo_view x = in, y;
    int i;

    for (i = 0; i < ch->n; i++) {
        y = i == ch->n - 1 ? out : conv_tmp(m, &ch->v[i], x);
        conv(m, &ch->v[i], x, y, NULL, nb);
        x = y;
    }
}

static struct yolo_view input_of(struct yolo_model *m, int i, int f)
{
    int src = f < 0 ? i + f : f;

    if (src < -1 || src >= i)
        G_fatal_error(_("Layer %d reads from layer %d"), i, src);
    if (src == -1) {
        struct yolo_view v;

        v.buf = m->input;
        v.off = 0;
        v.bs = 3L * YOLO_IMG * YOLO_IMG;
        v.c = 3;
        v.h = v.w = YOLO_IMG;
        return v;
    }
    return m->layers[src].out;
}

static void segment(struct yolo_model *m, int i, int nb)
{
    struct yolo_layer *L = &m->layers[i];
    struct yolo_head *h = &m->head;
    struct yolo_view x0, p1, p2, p3;
    int l, nbox = 4 * h->reg_max, ch = nbox + h->nc + h->nm;
    char what[64];

    for (l = 0; l < h->nl; l++) {
        struct yolo_view x = input_of(m, i, L->f[l]), o;

        snprintf(what, sizeof(what), "head level %d", l);
        o = persistent(m, &h->lvl[l], ch, x.h, x.w, what);
        h->stride[l] = YOLO_IMG / x.w;
        chain(m, &h->box[l], x, slice(o, 0, nbox), nb);
        chain(m, &h->cls[l], x, slice(o, nbox, h->nc), nb);
        chain(m, &h->mc[l], x, slice(o, nbox + h->nc, h->nm), nb);
    }
    x0 = input_of(m, i, L->f[0]);
    p1 = conv_tmp(m, &h->p_cv1, x0);
    conv(m, &h->p_cv1, x0, p1, NULL, nb);
    p2 = tmp(m, h->p_up.cout, 2 * p1.h, 2 * p1.w);
    conv(m, &h->p_up, p1, p2, NULL, nb);
    p3 = conv_tmp(m, &h->p_cv2, p2);
    conv(m, &h->p_cv2, p2, p3, NULL, nb);
    conv(m, &h->p_cv3, p3,
         persistent(m, &h->proto, h->nm, p3.h, p3.w, "mask prototypes"), NULL,
         nb);
}

static void forward_layers(struct yolo_model *m, int nb);

static void forward(struct yolo_model *m, int nb)
{
    forward_layers(m, nb);
#ifdef HAVE_BLAS
    if (!m->planning && host_convs(m))
        unmap_all(m);
#endif
}

static void forward_layers(struct yolo_model *m, int nb)
{
    int i, j;
    char what[64];

    for (i = 0; i < m->nlayers; i++) {
        struct yolo_layer *L = &m->layers[i];
        struct yolo_view in = input_of(m, i, L->f[0]);

        m->scratch_top = 0;
        snprintf(what, sizeof(what), "layer %d", i);
        switch (L->kind) {
        case YL_CONV:
            conv(m, &L->cv1, in,
                 persistent(m, &L->out, L->cv1.cout, conv_hw(&L->cv1, in.h),
                            conv_hw(&L->cv1, in.w), what),
                 NULL, nb);
            break;
        case YL_C2F:
            c2f(m, L, in, persistent(m, &L->out, L->cv2.cout, in.h, in.w, what),
                nb);
            break;
        case YL_SPPF:
            sppf(m, L, in,
                 persistent(m, &L->out, L->cv2.cout, in.h, in.w, what), nb);
            break;
        case YL_C2PSA:
            c2psa(m, L, in,
                  persistent(m, &L->out, L->cv2.cout, in.h, in.w, what), nb);
            break;
        case YL_UPSAMPLE:
            upsample(m, in,
                     persistent(m, &L->out, in.c, 2 * in.h, 2 * in.w, what),
                     nb);
            break;
        case YL_CONCAT: {
            int c = 0, c0 = 0;
            struct yolo_view o;

            for (j = 0; j < L->nf; j++) {
                struct yolo_view v = input_of(m, i, L->f[j]);

                if (v.h != in.h || v.w != in.w)
                    G_fatal_error(_("Concatenation of layer %d mixes sizes"),
                                  i);
                c += v.c;
            }
            o = persistent(m, &L->out, c, in.h, in.w, what);
            for (j = 0; j < L->nf; j++) {
                struct yolo_view v = input_of(m, i, L->f[j]);

                copy_view(m, v, slice(o, c0, v.c), nb);
                c0 += v.c;
            }
            break;
        }
        case YL_SEGMENT:
            segment(m, i, nb);
            break;
        }
    }
}

/* Input and head output buffers: in host-visible memory on unified
 * memory devices, so that mapping them needs no copy. */
static cl_mem alloc_io(struct yolo_model *m, size_t n, const char *what)
{
    cl_int err;
    cl_mem b;

    if (!m->ocl->unified)
        return ocl_alloc(m->ocl, n, what);
    b = clCreateBuffer(m->ocl->context,
                       CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, n, NULL,
                       &err);
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: allocating %lu MiB for <%s> failed: %s"),
                      (unsigned long)(n >> 20), what, ocl_errstr(err));
    return b;
}

void yolo_load(struct yolo_model *m, struct ocl_backend *ocl,
               const struct yolo_setup *setup, const struct pth_file *pf,
               int batch)
{
    char path[64];
    const char *cls;
    int i, l, s, have_head = 0;
    size_t wgs;
    double mib = 0;

    memset(m, 0, sizeof(*m));
    m->ocl = ocl;
    m->setup = *setup;
    m->esize = setup->half ? 2 : 4;
    m->batch = batch;
#ifdef HAVE_BLAS
    if (host_convs(m))
        blas_single_thread();
#endif

    if (pf->nmodules == 0 || !pth_module(pf, "model.0"))
        G_fatal_error(_("<%s> does not hold a whole Ultralytics model (a "
                        "Delineate-Anything .pt checkpoint is expected)"),
                      pf->path);
    for (m->nlayers = 0;; m->nlayers++) {
        snprintf(path, sizeof(path), "model.%d", m->nlayers);
        if (!pth_module(pf, path))
            break;
    }
    m->layers = G_calloc(m->nlayers, sizeof(struct yolo_layer));
    for (i = 0; i < m->nlayers; i++) {
        struct yolo_layer *L = &m->layers[i];

        snprintf(path, sizeof(path), "model.%d", i);
        cls = module_class(pf, path);
        layer_inputs(pf, path, L);
        if (have_head)
            G_fatal_error(_("Layers after the segmentation head"));
        if (strcmp(cls, "Conv") == 0) {
            L->kind = YL_CONV;
            load_conv(m, pf, path, &L->cv1);
        }
        else if (strcmp(cls, "C3k2") == 0 || strcmp(cls, "C2f") == 0) {
            L->kind = YL_C2F;
            load_c2f(m, pf, path, L);
        }
        else if (strcmp(cls, "SPPF") == 0) {
            char sub[80];

            L->kind = YL_SPPF;
            snprintf(sub, sizeof(sub), "%s.cv1", path);
            load_conv(m, pf, sub, &L->cv1);
            snprintf(sub, sizeof(sub), "%s.cv2", path);
            load_conv(m, pf, sub, &L->cv2);
            snprintf(sub, sizeof(sub), "%s.m", path);
            L->pool = (int)attr_int(pf, sub, "kernel_size", 5);
        }
        else if (strcmp(cls, "C2PSA") == 0) {
            L->kind = YL_C2PSA;
            load_c2psa(m, pf, path, L);
        }
        else if (strcmp(cls, "Upsample") == 0) {
            const struct pth_attr *mode =
                pth_attr(pth_module(pf, path), "mode");

            L->kind = YL_UPSAMPLE;
            if (attr_double(pf, path, "scale_factor", 0) != 2.0 || !mode ||
                mode->type != PA_STR || strcmp(mode->s, "nearest") != 0)
                G_fatal_error(_("Only 2x nearest upsampling is supported "
                                "(layer %d)"),
                              i);
        }
        else if (strcmp(cls, "Concat") == 0) {
            L->kind = YL_CONCAT;
            if (attr_int(pf, path, "d", 1) != 1)
                G_fatal_error(_("Only channel concatenation is supported "
                                "(layer %d)"),
                              i);
        }
        else if (strcmp(cls, "Segment") == 0) {
            L->kind = YL_SEGMENT;
            load_segment(m, pf, path);
            if (L->nf != m->head.nl)
                G_fatal_error(_("Head reads %d levels, %d expected"), L->nf,
                              m->head.nl);
            have_head = 1;
        }
        else if (strcmp(cls, "Detect") == 0)
            G_fatal_error(_("<%s> is a detection model: Delineate-Anything "
                            "needs a segmentation model"),
                          pf->path);
        else
            G_fatal_error(_("Layer %d of class %s is not supported"), i, cls);
    }
    if (!have_head)
        G_fatal_error(_("No segmentation head in <%s>"), pf->path);
    if (m->head.nc != 1)
        G_message(_("The model has %d classes; any class is taken as a "
                    "field"),
                  m->head.nc);

    m->k_gemm = ocl_kernel(ocl, "conv_gemm");
    m->k_dw = ocl_kernel(ocl, "dwconv3x3");
    m->k_pool = ocl_kernel(ocl, "maxpool");
    m->k_up = ocl_kernel(ocl, "upsample2");
    m->k_copy = ocl_kernel(ocl, "copy_planes");
    m->k_attn = ocl_kernel(ocl, "attention");
    ocl_check(clGetKernelWorkGroupInfo(m->k_gemm, ocl->device,
                                       CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs),
                                       &wgs, NULL),
              "clGetKernelWorkGroupInfo");
    if (wgs < WG * WG)
        G_fatal_error(_("OpenCL: device <%s> runs at most %lu work-items per "
                        "work-group for the convolution kernel, %d are "
                        "needed"),
                      ocl->device_name, (unsigned long)wgs, WG * WG);

    /* Dry run: allocate the layer outputs, size the scratch buffer. */
    m->nslots = setup->profile == YP_CPU ? 1 : 2;
    m->input = alloc_io(m, (size_t)m->esize * 3 * YOLO_IMG * YOLO_IMG * batch,
                        "network input");
    m->planning = 1;
    forward(m, batch);
    m->planning = 0;
    m->scratch = ocl_alloc(ocl, (size_t)m->esize * (m->scratch_peak + 1),
                           "scratch activations");
    m->xfer = setup->profile == YP_EGPU ? ocl_queue(ocl) : ocl->queue;

    for (i = 0; i < m->nlayers; i++)
        if (m->layers[i].out.buf)
            mib += m->layers[i].out.bs * batch * (double)m->esize / 1048576;
    mib += m->scratch_peak * (double)m->esize / 1048576;
    for (s = 0; s < m->nslots; s++) {
        struct yolo_slot *sl = &m->slot[s];
        size_t nin = 3L * YOLO_IMG * YOLO_IMG * batch;

        sl->input = s == 0
                        ? m->input
                        : alloc_io(m, (size_t)m->esize * nin, "network input");
        for (l = 0; l < m->head.nl; l++) {
            size_t n = (size_t)m->head.lvl[l].bs * batch;

            sl->lvl[l] = s == 0 ? m->head.lvl[l].buf
                                : alloc_io(m, n * m->esize, "head level");
            /* Host arrays unless fp32 outputs are read through mappings. */
            sl->own_out = m->esize == 2 || !ocl->unified;
            if (sl->own_out)
                sl->out_lvl[l] = G_malloc(sizeof(float) * n);
            if (m->esize == 2)
                sl->raw_lvl[l] = ocl->unified ? NULL : G_malloc(2 * n);
            else if (!ocl->unified)
                sl->raw_lvl[l] = sl->out_lvl[l];
            mib += n * (double)m->esize * (s ? 1 : 0) / 1048576;
        }
        {
            size_t n = (size_t)m->head.proto.bs * batch;

            sl->proto = s == 0 ? m->head.proto.buf
                               : alloc_io(m, n * m->esize, "mask prototypes");
            if (sl->own_out)
                sl->out_proto = G_malloc(sizeof(float) * n);
            if (m->esize == 2)
                sl->raw_proto = ocl->unified ? NULL : G_malloc(2 * n);
            else if (!ocl->unified)
                sl->raw_proto = sl->out_proto;
        }
        if (!ocl->unified) {
            cl_int err;

            /* Pinned staging memory for fast, asynchronous uploads. */
            sl->staging = clCreateBuffer(ocl->context, CL_MEM_ALLOC_HOST_PTR,
                                         (size_t)m->esize * nin, NULL, &err);
            ocl_check(err, "clCreateBuffer (staging)");
            sl->stage = clEnqueueMapBuffer(
                ocl->queue, sl->staging, CL_TRUE, CL_MAP_WRITE, 0,
                (size_t)m->esize * nin, 0, NULL, NULL, &err);
            ocl_check(err, "clEnqueueMapBuffer (staging)");
        }
        if (m->esize == 2)
            sl->in = G_malloc(sizeof(float) * nin);
    }
    G_verbose_message(_("Network: %d layers, %.1f GFLOP per tile, %.0f MiB "
                        "of activations for batch=%d, %d batch%s in flight"),
                      m->nlayers, m->gflop, mib, batch, m->nslots,
                      m->nslots > 1 ? "es" : "");
}

/* Conversions between float32 and float16 (round to nearest even). */

static uint16_t to_half(float f)
{
    uint32_t u, sign, e, mant;

    memcpy(&u, &f, 4);
    sign = (u >> 16) & 0x8000;
    e = (u >> 23) & 0xff;
    mant = u & 0x7fffff;
    if (e == 0xff)
        return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (e > 142)
        return (uint16_t)(sign | 0x7c00);
    if (e < 113) {
        uint32_t shift, h, rem, half;

        if (e < 102)
            return (uint16_t)sign;
        mant |= 0x800000;
        shift = 126 - e;
        h = mant >> shift;
        rem = mant & ((1u << shift) - 1);
        half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1)))
            h++;
        return (uint16_t)(sign | h);
    }
    {
        uint32_t h = ((e - 112) << 10) | (mant >> 13), rem = mant & 0x1fff;

        if (rem > 0x1000 || (rem == 0x1000 && (h & 1)))
            h++;
        return (uint16_t)(sign | h);
    }
}

static float from_half(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f, mant = h & 0x3ff, u;
    float f;

    if (e == 0) {
        if (mant == 0)
            u = sign;
        else {
            e = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                e--;
            }
            u = sign | (e << 23) | ((mant & 0x3ff) << 13);
        }
    }
    else if (e == 31)
        u = sign | 0x7f800000 | (mant << 13);
    else
        u = sign | ((e + 112) << 23) | (mant << 13);
    memcpy(&f, &u, 4);
    return f;
}

static void floats_to_half(const float *src, uint16_t *dst, size_t n)
{
    size_t i;

#pragma omp parallel for
    for (i = 0; i < n; i++)
        dst[i] = to_half(src[i]);
}

static void half_to_floats(const uint16_t *src, float *dst, size_t n)
{
    size_t i;

#pragma omp parallel for
    for (i = 0; i < n; i++)
        dst[i] = from_half(src[i]);
}

static void *map(struct yolo_model *m, cl_command_queue q, cl_mem b,
                 cl_map_flags fl, size_t n, cl_bool block, cl_event *ev)
{
    cl_int err;
    void *p = clEnqueueMapBuffer(q, b, block, fl, 0, n, 0, NULL, ev, &err);

    (void)m;
    ocl_check(err, "clEnqueueMapBuffer");
    return p;
}

float *yolo_input(struct yolo_model *m, int s)
{
    struct yolo_slot *sl = &m->slot[s];
    size_t n = (size_t)m->esize * 3 * YOLO_IMG * YOLO_IMG * m->batch;

    if (m->ocl->unified)
        /* Zero copy: the caller writes into the device buffer. */
        sl->stage = map(m, m->ocl->queue, sl->input,
                        strncmp(m->ocl->version, "OpenCL C 1.1", 12) == 0
                            ? CL_MAP_WRITE
                            : CL_MAP_WRITE_INVALIDATE_REGION,
                        n, CL_TRUE, NULL);
    return m->esize == 2 ? sl->in : (float *)sl->stage;
}

void yolo_submit(struct yolo_model *m, int s, int nb)
{
    struct yolo_slot *sl = &m->slot[s];
    struct ocl_backend *ocl = m->ocl;
    size_t nin = 3L * YOLO_IMG * YOLO_IMG * nb;
    cl_event ev = NULL;
    double t0 = ocl->profile ? wall() : 0;
    int l;

    if (nb < 1 || nb > m->batch)
        G_fatal_error(_("Invalid batch of %d images"), nb);
    sl->nb = nb;
    if (m->esize == 2)
        floats_to_half(sl->in, sl->stage, nin);
    if (ocl->unified)
        ocl_check(clEnqueueUnmapMemObject(ocl->queue, sl->input, sl->stage, 0,
                                          NULL, NULL),
                  "clEnqueueUnmapMemObject");
    else {
        ocl_check(clEnqueueWriteBuffer(m->xfer, sl->input, CL_FALSE, 0,
                                       nin * m->esize, sl->stage, 0, NULL, &ev),
                  "clEnqueueWriteBuffer");
        if (m->xfer != ocl->queue) {
            ocl_check(clEnqueueBarrierWithWaitList(ocl->queue, 1, &ev, NULL),
                      "clEnqueueBarrierWithWaitList");
            clFlush(m->xfer);
        }
        clReleaseEvent(ev);
    }
    if (ocl->profile) {
        clFinish(m->xfer);
        clFinish(ocl->queue);
        prof_add(m, "upload", wall() - t0, 0, (double)nin * m->esize);
    }

    m->input = sl->input;
    for (l = 0; l < m->head.nl; l++)
        m->head.lvl[l].buf = sl->lvl[l];
    m->head.proto.buf = sl->proto;
    forward(m, nb);

    /* Head outputs: mapped on unified memory, otherwise read back on the
     * transfer queue once the compute is done. */
    t0 = ocl->profile ? wall() : 0;
    if (ocl->unified) {
        for (l = 0; l <= m->head.nl; l++) {
            const struct yolo_view *v =
                l < m->head.nl ? &m->head.lvl[l] : &m->head.proto;
            void *p = map(m, ocl->queue, v->buf, CL_MAP_READ,
                          (size_t)m->esize * v->bs * nb, CL_FALSE,
                          l == m->head.nl ? &sl->done : NULL);

            if (l < m->head.nl)
                sl->raw_lvl[l] = p;
            else
                sl->raw_proto = p;
        }
    }
    else {
        cl_event ec = NULL;

        if (m->xfer != ocl->queue) {
            ocl_check(clEnqueueMarkerWithWaitList(ocl->queue, 0, NULL, &ec),
                      "clEnqueueMarkerWithWaitList");
            clFlush(ocl->queue);
        }
        for (l = 0; l <= m->head.nl; l++) {
            const struct yolo_view *v =
                l < m->head.nl ? &m->head.lvl[l] : &m->head.proto;

            ocl_check(
                clEnqueueReadBuffer(
                    m->xfer, v->buf, CL_FALSE, 0, (size_t)m->esize * v->bs * nb,
                    l < m->head.nl ? sl->raw_lvl[l] : sl->raw_proto, ec ? 1 : 0,
                    ec ? &ec : NULL, l == m->head.nl ? &sl->done : NULL),
                "clEnqueueReadBuffer");
        }
        if (ec)
            clReleaseEvent(ec);
    }
    clFlush(m->xfer);
    clFlush(ocl->queue);
    if (ocl->profile) {
        clWaitForEvents(1, &sl->done);
    }
    if (ocl->profile) {
        double bytes = (double)m->head.proto.bs;

        for (l = 0; l < m->head.nl; l++)
            bytes += m->head.lvl[l].bs;
        prof_add(m, "download", wall() - t0, 0, bytes * m->esize * nb);
    }
}

void yolo_wait(struct yolo_model *m, int s)
{
    struct yolo_slot *sl = &m->slot[s];
    int l;

    ocl_check(clWaitForEvents(1, &sl->done), "clWaitForEvents");
    clReleaseEvent(sl->done);
    sl->done = NULL;
    for (l = 0; l <= m->head.nl; l++) {
        const struct yolo_view *v =
            l < m->head.nl ? &m->head.lvl[l] : &m->head.proto;
        void *raw = l < m->head.nl ? sl->raw_lvl[l] : sl->raw_proto;
        float **out = l < m->head.nl ? &sl->out_lvl[l] : &sl->out_proto;

        if (m->esize == 2)
            half_to_floats(raw, *out, (size_t)v->bs * sl->nb);
        else if (m->ocl->unified)
            *out = raw;
    }
    /* Converted copies no longer need the mappings. */
    if (m->esize == 2 && m->ocl->unified)
        yolo_release(m, s);
}

void yolo_release(struct yolo_model *m, int s)
{
    struct yolo_slot *sl = &m->slot[s];
    int l;

    if (!m->ocl->unified)
        return;
    for (l = 0; l <= m->head.nl; l++) {
        cl_mem b = l < m->head.nl ? sl->lvl[l] : sl->proto;
        void **raw = l < m->head.nl ? &sl->raw_lvl[l] : &sl->raw_proto;

        if (*raw)
            ocl_check(
                clEnqueueUnmapMemObject(m->ocl->queue, b, *raw, 0, NULL, NULL),
                "clEnqueueUnmapMemObject");
        *raw = NULL;
    }
}

void yolo_report(const struct yolo_model *m)
{
    double total = 0;
    int i;

    for (i = 0; i < m->nprof; i++)
        total += m->prof[i].sec;
    fprintf(stderr, "%-12s %8s %10s %8s %10s %10s\n", "kernel", "calls",
            "seconds", "share", "GFLOP/s", "GB/s");
    for (i = 0; i < m->nprof; i++) {
        const struct yolo_prof *p = &m->prof[i];

        fprintf(stderr, "%-12s %8ld %10.3f %7.1f%% %10.2f %10.2f\n", p->name,
                p->calls, p->sec, total > 0 ? 100 * p->sec / total : 0,
                p->sec > 0 ? 1e-9 * p->flop / p->sec : 0,
                p->sec > 0 ? 1e-9 * p->bytes / p->sec : 0);
    }
}

/* Post-processing on the host. */

struct cand {
    float x1, y1, x2, y2, score;
    int lvl, idx; /* Head level and anchor index in that level. */
};

static int cmp_cand(const void *a, const void *b)
{
    const struct cand *x = a, *y = b;

    if (x->score != y->score)
        return x->score > y->score ? -1 : 1;
    if (x->lvl != y->lvl)
        return x->lvl - y->lvl;
    return x->idx - y->idx;
}

static float box_iou(const struct cand *a, const struct cand *b)
{
    float iw = fminf(a->x2, b->x2) - fmaxf(a->x1, b->x1);
    float ih = fminf(a->y2, b->y2) - fmaxf(a->y1, b->y1);
    float inter, ua;

    if (iw <= 0 || ih <= 0)
        return 0.0f;
    inter = iw * ih;
    ua = (a->x2 - a->x1) * (a->y2 - a->y1) + (b->x2 - b->x1) * (b->y2 - b->y1) -
         inter;
    return ua > 0 ? inter / ua : 0.0f;
}

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

/* PyTorch bilinear interpolation source index (align_corners=False). */
static void src_index(int dst, float scale, int in, int *i0, int *i1, float *l1)
{
    float s = scale * (dst + 0.5f) - 0.5f;

    if (s < 0)
        s = 0;
    *i0 = (int)s;
    *i1 = *i0 + (*i0 < in - 1 ? 1 : 0);
    *l1 = s - *i0;
}

/* One 3 x 3 erosion (erode = 1) or dilation of a window mask; cells
 * outside the window but inside the tile are 0, cells outside the tile
 * are ignored, as torch's max_pool2d pads with -inf. */
static void morph(const unsigned char *in, unsigned char *out, int x0, int y0,
                  int w, int h, int erode)
{
    int x, y, dx, dy;

    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            int v = erode ? 1 : 0;

            for (dy = -1; dy <= 1; dy++) {
                int yy = y + dy, ty = y0 + yy;

                if (ty < 0 || ty >= YOLO_IMG)
                    continue;
                for (dx = -1; dx <= 1; dx++) {
                    int xx = x + dx, tx = x0 + xx, c;

                    if (tx < 0 || tx >= YOLO_IMG)
                        continue;
                    c = xx >= 0 && xx < w && yy >= 0 && yy < h ? in[yy * w + xx]
                                                               : 0;
                    if (erode && !c)
                        v = 0;
                    if (!erode && c)
                        v = 1;
                }
            }
            out[y * w + x] = (unsigned char)v;
        }
}

void yolo_detect(const struct yolo_model *m, int s, int b, float conf,
                 float iou, int max_det, struct yolo_dets *out)
{
    const struct yolo_slot *sl = &m->slot[s];
    const struct yolo_head *h = &m->head;
    int nbox = 4 * h->reg_max, l, i, j, n = 0, cap = 256, nkeep = 0;
    struct cand *c = G_malloc(cap * sizeof(struct cand));
    char *sup;
    int pw = h->proto.w, ph = h->proto.h;
    const float *proto = sl->out_proto + (size_t)b * h->proto.bs;

    out->n = 0;
    /* Candidates: best class score above conf, boxes from the
     * distribution focal loss bins (softmax expectation per side). */
    for (l = 0; l < h->nl; l++) {
        int hw = h->lvl[l].h * h->lvl[l].w, s = h->stride[l];
        const float *o = sl->out_lvl[l] + (size_t)b * h->lvl[l].bs;

        for (i = 0; i < hw; i++) {
            float best = -INFINITY, d[4], score;
            int k, side;

            for (k = 0; k < h->nc; k++)
                best = fmaxf(best, o[(size_t)(nbox + k) * hw + i]);
            score = 1.0f / (1.0f + expf(-best));
            if (!(score > conf))
                continue;
            for (side = 0; side < 4; side++) {
                float mx = -INFINITY, sum = 0, e = 0;

                for (k = 0; k < h->reg_max; k++)
                    mx = fmaxf(mx, o[(size_t)(side * h->reg_max + k) * hw + i]);
                for (k = 0; k < h->reg_max; k++) {
                    float p =
                        expf(o[(size_t)(side * h->reg_max + k) * hw + i] - mx);

                    sum += p;
                    e += p * k;
                }
                d[side] = e / sum;
            }
            if (n == cap) {
                cap *= 2;
                c = G_realloc(c, cap * sizeof(struct cand));
            }
            {
                float ax = i % h->lvl[l].w + 0.5f, ay = i / h->lvl[l].w + 0.5f;
                float x1 = ax - d[0], y1 = ay - d[1], x2 = ax + d[2],
                      y2 = ay + d[3];
                float cx = (x1 + x2) / 2 * s, cy = (y1 + y2) / 2 * s;
                float bw = (x2 - x1) * s, bh = (y2 - y1) * s;

                c[n].x1 = cx - bw / 2;
                c[n].y1 = cy - bh / 2;
                c[n].x2 = cx + bw / 2;
                c[n].y2 = cy + bh / 2;
            }
            c[n].score = score;
            c[n].lvl = l;
            c[n].idx = i;
            n++;
        }
    }
    if (n > 30000) {
        qsort(c, n, sizeof(struct cand), cmp_cand);
        n = 30000;
    }

    /* Greedy non-maximum suppression in decreasing score order. */
    qsort(c, n, sizeof(struct cand), cmp_cand);
    sup = G_calloc(n ? n : 1, 1);
    for (i = 0; i < n && nkeep < max_det; i++) {
        if (sup[i])
            continue;
        c[nkeep++] = c[i];
        for (j = i + 1; j < n; j++)
            if (!sup[j] && box_iou(&c[i], &c[j]) > iou)
                sup[j] = 1;
    }
    G_free(sup);

    /* Masks: coefficients times prototypes, bilinear upsampling to the
     * tile, crop to the box, threshold at 0, over the box widened by 4
     * pixels. */
    for (i = 0; i < nkeep; i++) {
        const struct cand *k = &c[i];
        int hw = h->lvl[k->lvl].h * h->lvl[k->lvl].w;
        const float *o = sl->out_lvl[k->lvl] + (size_t)b * h->lvl[k->lvl].bs;
        float x1 = clampf(k->x1, 0, YOLO_IMG), y1 = clampf(k->y1, 0, YOLO_IMG);
        float x2 = clampf(k->x2, 0, YOLO_IMG), y2 = clampf(k->y2, 0, YOLO_IMG);
        float coef[64], sc = (float)pw / YOLO_IMG;
        int wx0, wy0, wx1, wy1, ww, wh, px0, px1, py0, py1, lw, x, y, q,
            any = 0;
        float *logit;
        unsigned char *mk, *t;
        struct yolo_det *d;

        wx0 = (int)x1 - 4 > 0 ? (int)x1 - 4 : 0;
        wy0 = (int)y1 - 4 > 0 ? (int)y1 - 4 : 0;
        wx1 = (int)ceilf(x2) + 4 < YOLO_IMG ? (int)ceilf(x2) + 4 : YOLO_IMG;
        wy1 = (int)ceilf(y2) + 4 < YOLO_IMG ? (int)ceilf(y2) + 4 : YOLO_IMG;
        ww = wx1 - wx0;
        wh = wy1 - wy0;
        if (ww <= 0 || wh <= 0)
            continue;
        for (q = 0; q < h->nm && q < 64; q++)
            coef[q] = o[(size_t)(nbox + h->nc + q) * hw + k->idx];

        /* Prototype cells the window samples. */
        {
            int a0, a1;
            float lam;

            src_index(wx0, sc, pw, &px0, &a1, &lam);
            src_index(wx1 - 1, sc, pw, &a0, &px1, &lam);
            src_index(wy0, sc, ph, &py0, &a1, &lam);
            src_index(wy1 - 1, sc, ph, &a0, &py1, &lam);
        }
        lw = px1 - px0 + 1;
        logit = G_malloc(sizeof(float) * lw * (py1 - py0 + 1));
        for (y = py0; y <= py1; y++)
            for (x = px0; x <= px1; x++) {
                float v = 0;

                for (q = 0; q < h->nm; q++)
                    v += coef[q] * proto[((size_t)q * ph + y) * pw + x];
                logit[(y - py0) * lw + x - px0] = v;
            }
        mk = G_malloc((size_t)ww * wh);
        for (y = 0; y < wh; y++) {
            int ty = wy0 + y, y0i, y1i;
            float ly;

            src_index(ty, sc, ph, &y0i, &y1i, &ly);
            for (x = 0; x < ww; x++) {
                int tx = wx0 + x, x0i, x1i;
                float lx, v;

                src_index(tx, sc, pw, &x0i, &x1i, &lx);
                v = (1 - ly) * ((1 - lx) * logit[(y0i - py0) * lw + x0i - px0] +
                                lx * logit[(y0i - py0) * lw + x1i - px0]) +
                    ly * ((1 - lx) * logit[(y1i - py0) * lw + x0i - px0] +
                          lx * logit[(y1i - py0) * lw + x1i - px0]);
                mk[y * ww + x] =
                    tx >= x1 && tx < x2 && ty >= y1 && ty < y2 && v > 0.0f;
                any |= mk[y * ww + x];
            }
        }
        G_free(logit);
        if (!any) {
            G_free(mk);
            continue;
        }

        /* Opening then closing with a 3 x 3 square. */
        t = G_malloc((size_t)ww * wh);
        morph(mk, t, wx0, wy0, ww, wh, 1);
        morph(t, mk, wx0, wy0, ww, wh, 0);
        morph(mk, t, wx0, wy0, ww, wh, 0);
        morph(t, mk, wx0, wy0, ww, wh, 1);
        G_free(t);

        if (out->n == out->cap) {
            out->cap = out->cap ? 2 * out->cap : 64;
            out->v = G_realloc(out->v, out->cap * sizeof(struct yolo_det));
        }
        d = &out->v[out->n++];
        d->x1 = x1;
        d->y1 = y1;
        d->x2 = x2;
        d->y2 = y2;
        d->score = k->score;
        d->wx0 = wx0;
        d->wy0 = wy0;
        d->ww = ww;
        d->wh = wh;
        d->mask = mk;
    }
    G_free(c);
}

void yolo_dets_free(struct yolo_dets *d)
{
    int i;

    for (i = 0; i < d->n; i++)
        G_free(d->v[i].mask);
    G_free(d->v);
    memset(d, 0, sizeof(*d));
}

static void dump_view(struct yolo_model *m, const struct yolo_view *v,
                      const char *dir, const char *name, FILE *index)
{
    char path[GPATH_MAX];
    size_t n = (size_t)v->c * v->h * v->w;
    float *buf = G_malloc(sizeof(float) * n);
    FILE *fp;

    if (m->esize == 2) {
        uint16_t *raw = G_malloc(2 * n);

        ocl_read(m->ocl, v->buf, 2 * v->off, raw, 2 * n);
        half_to_floats(raw, buf, n);
        G_free(raw);
    }
    else
        ocl_read(m->ocl, v->buf, sizeof(float) * v->off, buf,
                 sizeof(float) * n);
    snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
    fp = fopen(path, "wb");
    if (!fp || fwrite(buf, sizeof(float), n, fp) != n)
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    fclose(fp);
    fprintf(index, "%s %d %d %d\n", name, v->c, v->h, v->w);
    G_free(buf);
}

void yolo_dump(struct yolo_model *m, const char *dir)
{
    char path[GPATH_MAX], name[32];
    FILE *index;
    int i;

    snprintf(path, sizeof(path), "%s/layers.txt", dir);
    index = fopen(path, "w");
    if (!index)
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    {
        struct yolo_view in = input_of(m, 0, -1);

        dump_view(m, &in, dir, "input", index);
    }
    for (i = 0; i < m->nlayers; i++)
        if (m->layers[i].out.buf) {
            snprintf(name, sizeof(name), "layer_%02d", i);
            dump_view(m, &m->layers[i].out, dir, name, index);
        }
    for (i = 0; i < m->head.nl; i++) {
        snprintf(name, sizeof(name), "level_%d", i);
        dump_view(m, &m->head.lvl[i], dir, name, index);
    }
    dump_view(m, &m->head.proto, dir, "proto", index);
    fclose(index);
}

static void free_conv(struct yolo_conv *c)
{
    ocl_release(&c->w);
    ocl_release(&c->b);
    G_free(c->hw);
    G_free(c->hb);
}

static void free_bottleneck(struct yolo_bottleneck *b)
{
    free_conv(&b->cv1);
    free_conv(&b->cv2);
}

void yolo_free(struct yolo_model *m)
{
    int i, j, l;

    for (i = 0; i < m->nlayers; i++) {
        struct yolo_layer *L = &m->layers[i];

        free_conv(&L->cv1);
        free_conv(&L->cv2);
        if (L->kind == YL_C2F) {
            for (j = 0; j < L->n; j++) {
                struct yolo_item *it = &L->items[j];
                int k;

                free_bottleneck(&it->b);
                free_conv(&it->cv1);
                free_conv(&it->cv2);
                free_conv(&it->cv3);
                for (k = 0; k < it->nb; k++)
                    free_bottleneck(&it->m[k]);
                G_free(it->m);
            }
            G_free(L->items);
        }
        if (L->kind == YL_C2PSA) {
            for (j = 0; j < L->n; j++) {
                free_conv(&L->psa[j].qkv);
                free_conv(&L->psa[j].pe);
                free_conv(&L->psa[j].proj);
                free_conv(&L->psa[j].ffn0);
                free_conv(&L->psa[j].ffn1);
            }
            G_free(L->psa);
        }
        ocl_release(&L->out.buf);
    }
    G_free(m->layers);
    for (l = 0; l < m->head.nl; l++) {
        struct yolo_chain *chs[3] = {&m->head.box[l], &m->head.cls[l],
                                     &m->head.mc[l]};

        for (j = 0; j < 3; j++) {
            for (i = 0; i < chs[j]->n; i++)
                free_conv(&chs[j]->v[i]);
            G_free(chs[j]->v);
        }
    }
    free_conv(&m->head.p_cv1);
    free_conv(&m->head.p_up);
    free_conv(&m->head.p_cv2);
    free_conv(&m->head.p_cv3);
    /* The head outputs and input are those of the slots. */
    for (i = 0; i < m->nslots; i++) {
        struct yolo_slot *sl = &m->slot[i];

        yolo_release(m, i);
        for (l = 0; l < m->head.nl; l++) {
            if (sl->own_out)
                G_free(sl->out_lvl[l]);
            if (m->esize == 2 && !m->ocl->unified)
                G_free(sl->raw_lvl[l]);
            ocl_release(&sl->lvl[l]);
        }
        if (sl->own_out)
            G_free(sl->out_proto);
        if (m->esize == 2 && !m->ocl->unified)
            G_free(sl->raw_proto);
        ocl_release(&sl->proto);
        if (sl->staging) {
            clEnqueueUnmapMemObject(m->ocl->queue, sl->staging, sl->stage, 0,
                                    NULL, NULL);
            clFinish(m->ocl->queue);
            ocl_release(&sl->staging);
        }
        ocl_release(&sl->input);
        G_free(sl->in);
    }
    if (m->xfer && m->xfer != m->ocl->queue)
        clReleaseCommandQueue(m->xfer);
    ocl_release(&m->scratch);
    if (m->k_gemm) {
        clReleaseKernel(m->k_gemm);
        clReleaseKernel(m->k_dw);
        clReleaseKernel(m->k_pool);
        clReleaseKernel(m->k_up);
        clReleaseKernel(m->k_copy);
        clReleaseKernel(m->k_attn);
    }
    memset(m, 0, sizeof(*m));
}
