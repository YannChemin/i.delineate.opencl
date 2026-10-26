/****************************************************************************
 *
 * MODULE:       i.delineate.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Conversion of the field identity raster into a topological
 *               vector area map: Delineate-Anything's polygon filtering
 *               (small holes filled, small fields dropped), boundary
 *               tracing, simplification and topology cleaning.
 *
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 *****************************************************************************/

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <grass/dbmi.h>
#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>
#include <grass/vector.h>

#include "progress.h"
#include "vectorize.h"

/* Field value of a cell: an identity >= 2, minus a land cover class, or 0
 * where there is no field. */
static int key_of(int v) { return v >= 2 || v < 0 ? v : 0; }

/* 4-connected components of equal key; returns the component count and
 * fills comp with the component of every cell. */
static int label_regions(const int *ids, int rows, int cols, int *comp,
                         int **ckey)
{
    size_t n = (size_t)rows * cols, i, sp, scap = 1 << 16;
    int *stack = G_malloc(scap * sizeof(int));
    int nc = 0, cap = 1024;

    *ckey = G_malloc(cap * sizeof(int));
    for (i = 0; i < n; i++)
        comp[i] = -1;

    for (i = 0; i < n; i++) {
        int k = key_of(ids[i]);

        if (comp[i] >= 0)
            continue;
        if (nc == INT32_MAX)
            G_fatal_error(_("Too many regions"));
        if (nc == cap) {
            cap *= 2;
            *ckey = G_realloc(*ckey, cap * sizeof(int));
        }
        (*ckey)[nc] = k;
        comp[i] = nc;
        sp = 0;
        stack[sp++] = (int)i;
        while (sp) {
            size_t c = (size_t)stack[--sp];
            int r = (int)(c / cols), cc = (int)(c % cols), j;
            long nb[4];

            nb[0] = cc > 0 ? (long)c - 1 : -1;
            nb[1] = cc < cols - 1 ? (long)c + 1 : -1;
            nb[2] = r > 0 ? (long)c - cols : -1;
            nb[3] = r < rows - 1 ? (long)c + cols : -1;
            for (j = 0; j < 4; j++)
                if (nb[j] >= 0 && comp[nb[j]] < 0 && key_of(ids[nb[j]]) == k) {
                    comp[nb[j]] = nc;
                    if (sp == scap) {
                        scap *= 2;
                        stack = G_realloc(stack, scap * sizeof(int));
                    }
                    stack[sp++] = (int)nb[j];
                }
        }
        nc++;
    }
    G_free(stack);
    return nc;
}

/* Set of unordered component pairs. */
struct pairset {
    uint64_t *key;
    size_t cap, n;
};

static void pairset_add(struct pairset *s, uint64_t k);

static void pairset_grow(struct pairset *s)
{
    struct pairset g;
    size_t i;

    g.cap = s->cap ? 2 * s->cap : 1 << 16;
    g.n = 0;
    g.key = G_malloc(g.cap * sizeof(uint64_t));
    for (i = 0; i < g.cap; i++)
        g.key[i] = UINT64_MAX;
    for (i = 0; i < s->cap; i++)
        if (s->key[i] != UINT64_MAX)
            pairset_add(&g, s->key[i]);
    G_free(s->key);
    *s = g;
}

static void pairset_add(struct pairset *s, uint64_t k)
{
    size_t i;

    if (2 * (s->n + 1) > s->cap)
        pairset_grow(s);
    i = (size_t)((k * 0x9E3779B97F4A7C15ULL) >> 20) & (s->cap - 1);
    while (s->key[i] != UINT64_MAX) {
        if (s->key[i] == k)
            return;
        i = (i + 1) & (s->cap - 1);
    }
    s->key[i] = k;
    s->n++;
}

static int uf_find(int *parent, int a)
{
    while (parent[a] != a) {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    return a;
}

struct by_area {
    int c;
    double a;
};

static int cmp_area_desc(const void *a, const void *b)
{
    const struct by_area *x = a, *y = b;

    if (x->a != y->a)
        return x->a > y->a ? -1 : 1;
    return x->c - y->c;
}

/* Per-field information, indexed by category. */
struct field_rec {
    int key;
    double area;   /* Square meters. */
    int best_cell; /* Pole of inaccessibility (chamfer distance). */
    int best_dist;
};

/* Fill the small holes of the fields and drop the small fields, as
 * Delineate-Anything's polygon filtering does with its polygons (holes
 * below the hole threshold removed, then polygons below the area
 * threshold dropped). A hole is a connected set of other regions enclosed
 * by one field and not touching the raster border. Filling it absorbs the
 * empty cells and the fields too small to be kept; a field large enough
 * to be kept stays an island (the reference keeps it as a polygon
 * overlapping the filled one). Returns the category of every component (0 for
 * none) and fills *info (indexed by category), *ncat. */
static int *filter_fields(const int *comp, int rows, int cols, int nc,
                          const int *ckey, const struct vect_params *vp,
                          struct field_rec **info, int *ncat)
{
    int *parent = G_malloc(nc * sizeof(int)),
        *next = G_malloc(nc * sizeof(int));
    int *tail = G_malloc(nc * sizeof(int)), *stamp = G_malloc(nc * sizeof(int));
    int *catof = G_calloc(nc, sizeof(int)), *queue = G_malloc(nc * sizeof(int));
    unsigned char *border = G_calloc(nc, 1);
    double *area = G_calloc(nc, sizeof(double));
    int *adj_start, *adj_nb, r, c, i, nfilled = 0, ndropped = 0;
    struct pairset ps = {NULL, 0, 0};
    struct by_area *order;
    size_t k;

    /* Areas, border contact and adjacency of the components. */
    G_begin_cell_area_calculations();
    for (r = 0; r < rows; r++) {
        double ca = G_area_of_cell_at_row(r / vp->scale) /
                    ((double)vp->scale * vp->scale);

        for (c = 0; c < cols; c++) {
            size_t i0 = (size_t)r * cols + c;
            int a = comp[i0];

            area[a] += ca;
            if (r == 0 || c == 0 || r == rows - 1 || c == cols - 1)
                border[a] = 1;
            if (c + 1 < cols && comp[i0 + 1] != a) {
                int b = comp[i0 + 1];

                pairset_add(&ps, a < b ? ((uint64_t)a << 32) | (uint32_t)b
                                       : ((uint64_t)b << 32) | (uint32_t)a);
            }
            if (r + 1 < rows && comp[i0 + cols] != a) {
                int b = comp[i0 + cols];

                pairset_add(&ps, a < b ? ((uint64_t)a << 32) | (uint32_t)b
                                       : ((uint64_t)b << 32) | (uint32_t)a);
            }
        }
    }
    adj_start = G_calloc(nc + 1, sizeof(int));
    for (k = 0; k < ps.cap; k++)
        if (ps.key[k] != UINT64_MAX) {
            adj_start[(int)(ps.key[k] >> 32) + 1]++;
            adj_start[(int)(ps.key[k] & 0xffffffffu) + 1]++;
        }
    for (c = 0; c < nc; c++)
        adj_start[c + 1] += adj_start[c];
    adj_nb = G_malloc((adj_start[nc] ? adj_start[nc] : 1) * sizeof(int));
    {
        int *fill = G_malloc((nc ? nc : 1) * sizeof(int));

        memcpy(fill, adj_start, nc * sizeof(int));
        for (k = 0; k < ps.cap; k++)
            if (ps.key[k] != UINT64_MAX) {
                int a = (int)(ps.key[k] >> 32),
                    b = (int)(ps.key[k] & 0xffffffffu);

                adj_nb[fill[a]++] = b;
                adj_nb[fill[b]++] = a;
            }
        G_free(fill);
    }
    G_free(ps.key);

    for (c = 0; c < nc; c++) {
        parent[c] = c;
        next[c] = -1;
        tail[c] = c;
        stamp[c] = -1;
    }

    /* Holes, largest fields first. */
    order = G_malloc((nc ? nc : 1) * sizeof(struct by_area));
    k = 0;
    for (c = 0; c < nc; c++)
        if (ckey[c] != 0) {
            order[k].c = c;
            order[k++].a = area[c];
        }
    qsort(order, k, sizeof(struct by_area), cmp_area_desc);
    for (i = 0; i < (int)k; i++) {
        int f = order[i].c, m, j;
        double thresh = ckey[f] < 0 ? vp->bg_min_hole : vp->min_hole;

        if (uf_find(parent, f) != f || thresh <= 0)
            continue;
        stamp[f] = f;
        for (m = f; m >= 0; m = next[m])
            for (j = adj_start[m]; j < adj_start[m + 1]; j++) {
                int s0 = uf_find(parent, adj_nb[j]), qh = 0, qt = 0, ok = 1;
                double ha;

                if (s0 == f || stamp[s0] == f)
                    continue;
                /* The connected set of other regions around s0, without
                 * crossing f: a hole if small and away from the border. */
                stamp[s0] = f;
                queue[qt++] = s0;
                ha = area[s0];
                ok = !border[s0] && ha < thresh;
                while (ok && qh < qt) {
                    int u = queue[qh++], mm, jj;

                    for (mm = u; ok && mm >= 0; mm = next[mm])
                        for (jj = adj_start[mm]; ok && jj < adj_start[mm + 1];
                             jj++) {
                            int v = uf_find(parent, adj_nb[jj]);

                            if (v == f || stamp[v] == f)
                                continue;
                            stamp[v] = f;
                            queue[qt++] = v;
                            ha += area[v];
                            ok = !border[v] && ha < thresh;
                        }
                }
                if (!ok)
                    continue;
                for (qh = 0; qh < qt; qh++) {
                    int u = queue[qh];

                    if (ckey[u] != 0 &&
                        area[u] >=
                            (ckey[u] < 0 ? vp->bg_min_area : vp->min_area))
                        continue;
                    parent[u] = f;
                    next[tail[f]] = u;
                    tail[f] = tail[u];
                    area[f] += area[u];
                }
                nfilled++;
            }
    }
    G_free(order);

    /* Small fields are dropped. */
    *info = G_calloc(1, sizeof(struct field_rec));
    *ncat = 0;
    for (c = 0; c < nc; c++) {
        double thresh;

        if (uf_find(parent, c) != c || ckey[c] == 0)
            continue;
        thresh = ckey[c] < 0 ? vp->bg_min_area : vp->min_area;
        if (area[c] < thresh) {
            ndropped++;
            continue;
        }
        catof[c] = ++*ncat;
        *info = G_realloc(*info, (*ncat + 1) * sizeof(struct field_rec));
        (*info)[*ncat].key = ckey[c];
        (*info)[*ncat].area = area[c];
        (*info)[*ncat].best_dist = -1;
        (*info)[*ncat].best_cell = -1;
    }
    for (c = 0; c < nc; c++)
        catof[c] = catof[uf_find(parent, c)];
    G_verbose_message(_("%d holes filled, %d small fields dropped"), nfilled,
                      ndropped);

    G_free(parent);
    G_free(next);
    G_free(tail);
    G_free(stamp);
    G_free(queue);
    G_free(border);
    G_free(area);
    G_free(adj_start);
    G_free(adj_nb);
    return catof;
}

/* Chamfer (3-4) distance to the field border, computed in d; picks for
 * every field the cell farthest inside it, a centroid location that
 * survives boundary simplification. */
static void find_poles(const int *K, int *D, int rows, int cols,
                       struct field_rec *info)
{
    size_t n = (size_t)rows * cols, i;
    int r, c;

    for (r = 0; r < rows; r++)
        for (c = 0; c < cols; c++) {
            size_t i0 = (size_t)r * cols + c;
            int k = K[i0], border;

            border = r == 0 || c == 0 || r == rows - 1 || c == cols - 1 ||
                     K[i0 - 1] != k || K[i0 + 1] != k || K[i0 - cols] != k ||
                     K[i0 + cols] != k;
            D[i0] = border ? 3 : INT32_MAX / 2;
        }

#define RELAX(nr, nc, w)                                                       \
    do {                                                                       \
        if ((nr) >= 0 && (nr) < rows && (nc) >= 0 && (nc) < cols) {            \
            size_t j = (size_t)(nr) * cols + (nc);                             \
            if (K[j] == K[i0] && D[j] + (w) < D[i0])                           \
                D[i0] = D[j] + (w);                                            \
        }                                                                      \
    } while (0)

    for (r = 0; r < rows; r++)
        for (c = 0; c < cols; c++) {
            size_t i0 = (size_t)r * cols + c;

            RELAX(r, c - 1, 3);
            RELAX(r - 1, c, 3);
            RELAX(r - 1, c - 1, 4);
            RELAX(r - 1, c + 1, 4);
        }
    for (r = rows - 1; r >= 0; r--)
        for (c = cols - 1; c >= 0; c--) {
            size_t i0 = (size_t)r * cols + c;

            RELAX(r, c + 1, 3);
            RELAX(r + 1, c, 3);
            RELAX(r + 1, c + 1, 4);
            RELAX(r + 1, c - 1, 4);
        }
#undef RELAX

    for (i = 0; i < n; i++) {
        int k = K[i];

        if (k > 0 && D[i] > info[k].best_dist) {
            info[k].best_dist = D[i];
            info[k].best_cell = (int)i;
        }
    }
}

/* Boundary tracing on the cell-corner lattice. Vertex (r, c) is the
 * corner at row line r, column line c. Directions: 0 east, 1 south,
 * 2 west, 3 north. An edge exists where the categories on its two sides
 * differ (outside the region counts as 0). */

struct tracer {
    const int *K;
    int rows, cols;
    unsigned char *hvis, *vvis;
};

static int key_at(const struct tracer *t, int r, int c)
{
    if (r < 0 || c < 0 || r >= t->rows || c >= t->cols)
        return 0;
    return t->K[(size_t)r * t->cols + c];
}

/* Returns 1 if the edge leaving (r, c) in direction d exists, and
 * points *vis at its visited flag. */
static int edge(const struct tracer *t, int r, int c, int d,
                unsigned char **vis)
{
    switch (d) {
    case 0:
        if (c >= t->cols)
            return 0;
        *vis = t->hvis + (size_t)r * t->cols + c;
        return key_at(t, r - 1, c) != key_at(t, r, c);
    case 2:
        if (c <= 0)
            return 0;
        *vis = t->hvis + (size_t)r * t->cols + c - 1;
        return key_at(t, r - 1, c - 1) != key_at(t, r, c - 1);
    case 1:
        if (r >= t->rows)
            return 0;
        *vis = t->vvis + (size_t)r * (t->cols + 1) + c;
        return key_at(t, r, c - 1) != key_at(t, r, c);
    default:
        if (r <= 0)
            return 0;
        *vis = t->vvis + (size_t)(r - 1) * (t->cols + 1) + c;
        return key_at(t, r - 1, c - 1) != key_at(t, r - 1, c);
    }
}

static int degree(const struct tracer *t, int r, int c)
{
    unsigned char *v;
    int d, n = 0;

    for (d = 0; d < 4; d++)
        n += edge(t, r, c, d, &v);
    return n;
}

static const int DR[4] = {0, 1, 0, -1}, DC[4] = {1, 0, -1, 0};

struct chain {
    int n, cap;
    int *rc; /* Vertex row, column pairs. */
};

static void chain_add(struct chain *ch, int r, int c)
{
    if (ch->n == ch->cap) {
        ch->cap = ch->cap ? 2 * ch->cap : 64;
        ch->rc = G_realloc(ch->rc, 2 * ch->cap * sizeof(int));
    }
    ch->rc[2 * ch->n] = r;
    ch->rc[2 * ch->n + 1] = c;
    ch->n++;
}

/* Follow edges from (r0, c0) heading d until a node (degree != 2) or
 * back to the start, keeping only the turning vertices. */
static void walk(const struct tracer *t, int r0, int c0, int d,
                 struct chain *ch)
{
    int r = r0, c = c0;
    unsigned char *vis;

    ch->n = 0;
    chain_add(ch, r, c);
    for (;;) {
        int nd = -1, k;

        edge(t, r, c, d, &vis);
        *vis = 1;
        r += DR[d];
        c += DC[d];
        if ((r == r0 && c == c0) || degree(t, r, c) != 2) {
            chain_add(ch, r, c);
            return;
        }
        for (k = 0; k < 4; k++) {
            if (k == (d + 2) % 4)
                continue;
            if (edge(t, r, c, k, &vis) && !*vis) {
                nd = k;
                break;
            }
        }
        if (nd < 0) {
            chain_add(ch, r, c);
            return;
        }
        if (nd != d)
            chain_add(ch, r, c);
        d = nd;
    }
}

/* Douglas-Peucker simplification of a polyline in map coordinates. */

static double seg_dist2(const double *p, const double *a, const double *b)
{
    double dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy, u;
    double ex, ey;

    u = l2 > 0 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2 : 0.0;
    if (u < 0)
        u = 0;
    else if (u > 1)
        u = 1;
    ex = a[0] + u * dx - p[0];
    ey = a[1] + u * dy - p[1];
    return ex * ex + ey * ey;
}

static void dp_rec(const double *xy, int a, int b, double tol2, char *keep)
{
    double dmax = 0.0;
    int i, imax = -1;

    for (i = a + 1; i < b; i++) {
        double d = seg_dist2(xy + 2 * i, xy + 2 * a, xy + 2 * b);

        if (d > dmax) {
            dmax = d;
            imax = i;
        }
    }
    if (imax >= 0 && dmax > tol2) {
        keep[imax] = 1;
        dp_rec(xy, a, imax, tol2, keep);
        dp_rec(xy, imax, b, tol2, keep);
    }
}

/* Simplify xy in place; returns the new point count. Closed rings are
 * split at their farthest vertex and never reduced below a triangle. */
static int simplify(double *xy, int n, double tol)
{
    char *keep;
    int i, m = 0, closed;

    if (tol <= 0 || n <= 2)
        return n;
    closed = xy[0] == xy[2 * (n - 1)] && xy[1] == xy[2 * (n - 1) + 1];
    if (closed && n < 5)
        return n;

    keep = G_calloc(n, 1);
    keep[0] = keep[n - 1] = 1;
    if (closed) {
        double dmax = -1.0;
        int f = 1;

        for (i = 1; i < n - 1; i++) {
            double dx = xy[2 * i] - xy[0], dy = xy[2 * i + 1] - xy[1];

            if (dx * dx + dy * dy > dmax) {
                dmax = dx * dx + dy * dy;
                f = i;
            }
        }
        keep[f] = 1;
        dp_rec(xy, 0, f, tol * tol, keep);
        dp_rec(xy, f, n - 1, tol * tol, keep);
    }
    else
        dp_rec(xy, 0, n - 1, tol * tol, keep);

    for (i = 0; i < n; i++)
        m += keep[i];
    if (closed && m < 4) {
        G_free(keep);
        return n;
    }
    for (i = 0, m = 0; i < n; i++)
        if (keep[i]) {
            xy[2 * m] = xy[2 * i];
            xy[2 * m + 1] = xy[2 * i + 1];
            m++;
        }
    G_free(keep);
    return m;
}

static void write_boundaries(struct Map_info *Map, const int *K, int rows,
                             int cols, const struct Cell_head *win, double tol)
{
    struct tracer t = {K, rows, cols, NULL, NULL};
    struct chain ch = {0, 0, NULL};
    struct line_pnts *Points = Vect_new_line_struct();
    struct line_cats *Cats = Vect_new_cats_struct();
    double *xy = NULL;
    int xycap = 0, r, c, d, nlines = 0;
    unsigned char *vis;

    t.hvis = G_calloc((size_t)(rows + 1) * cols, 1);
    t.vvis = G_calloc((size_t)rows * (cols + 1), 1);

#define EMIT()                                                                 \
    do {                                                                       \
        int k_, n_;                                                            \
        if (ch.n > xycap) {                                                    \
            xycap = ch.n;                                                      \
            xy = G_realloc(xy, 2 * xycap * sizeof(double));                    \
        }                                                                      \
        for (k_ = 0; k_ < ch.n; k_++) {                                        \
            xy[2 * k_] = win->west + ch.rc[2 * k_ + 1] * win->ew_res;          \
            xy[2 * k_ + 1] = win->north - ch.rc[2 * k_] * win->ns_res;         \
        }                                                                      \
        n_ = simplify(xy, ch.n, tol);                                          \
        Vect_reset_line(Points);                                               \
        for (k_ = 0; k_ < n_; k_++)                                            \
            Vect_append_point(Points, xy[2 * k_], xy[2 * k_ + 1], 0.0);        \
        Vect_write_line(Map, GV_BOUNDARY, Points, Cats);                       \
        nlines++;                                                              \
    } while (0)

    /* Chains between nodes. */
    for (r = 0; r <= rows; r++) {
        progress_part(0.4 + 0.4 * r / (rows + 1), 1);
        for (c = 0; c <= cols; c++) {
            int deg = degree(&t, r, c);

            if (deg == 0 || deg == 2)
                continue;
            for (d = 0; d < 4; d++) {
                if (edge(&t, r, c, d, &vis) && !*vis) {
                    walk(&t, r, c, d, &ch);
                    EMIT();
                }
            }
        }
    }

    /* Closed rings without any node. Every ring has a horizontal edge. */
    for (r = 0; r <= rows; r++)
        for (c = 0; c < cols; c++)
            if (edge(&t, r, c, 0, &vis) && !*vis) {
                walk(&t, r, c, 0, &ch);
                EMIT();
            }
#undef EMIT

    G_verbose_message(_("%d boundaries traced"), nlines);
    G_free(xy);
    G_free(ch.rc);
    G_free(t.hvis);
    G_free(t.vvis);
    Vect_destroy_line_struct(Points);
    Vect_destroy_cats_struct(Cats);
}

static void write_field_raster(const char *name, const int *K, int rows,
                               int cols, const struct Cell_head *fine,
                               const char *source)
{
    struct History hist;
    struct Cell_head out = *fine;
    CELL *buf;
    int fd, r, c;

    Rast_set_output_window(&out);
    buf = Rast_allocate_c_output_buf();
    fd = Rast_open_new(name, CELL_TYPE);
    for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++) {
            int k = K[(size_t)r * cols + c];

            if (k > 0)
                buf[c] = k;
            else
                Rast_set_c_null_value(&buf[c], 1);
        }
        Rast_put_c_row(fd, buf);
    }
    Rast_close(fd);
    G_free(buf);

    Rast_short_history(name, "raster", &hist);
    Rast_set_history(&hist, HIST_DATSRC_1, source);
    Rast_command_history(&hist);
    Rast_write_history(name, &hist);
}

static void write_attributes(struct Map_info *Map, const struct field_rec *info)
{
    struct field_info *Fi;
    dbDriver *driver;
    dbString sql;
    struct line_pnts *Points = Vect_new_line_struct();
    struct line_cats *Cats = Vect_new_cats_struct();
    char buf[1024];
    int line, nlines, cat;

    Fi = Vect_default_field_info(Map, 1, NULL, GV_1TABLE);
    Vect_map_add_dblink(Map, 1, NULL, Fi->table, GV_KEY_COLUMN, Fi->database,
                        Fi->driver);
    driver = db_start_driver_open_database(Fi->driver,
                                           Vect_subst_var(Fi->database, Map));
    if (!driver)
        G_fatal_error(_("Unable to open database <%s> by driver <%s>"),
                      Fi->database, Fi->driver);
    db_set_error_handler_driver(driver);

    db_init_string(&sql);
    snprintf(buf, sizeof(buf),
             "create table %s (%s integer, background integer, "
             "class integer, area double precision)",
             Fi->table, GV_KEY_COLUMN);
    db_set_string(&sql, buf);
    if (db_execute_immediate(driver, &sql) != DB_OK)
        G_fatal_error(_("Unable to create table: %s"), db_get_string(&sql));
    if (db_create_index2(driver, Fi->table, GV_KEY_COLUMN) != DB_OK)
        G_warning(_("Unable to create index"));
    if (db_grant_on_table(driver, Fi->table, DB_PRIV_SELECT,
                          DB_GROUP | DB_PUBLIC) != DB_OK)
        G_fatal_error(_("Unable to grant privileges on table <%s>"), Fi->table);

    db_begin_transaction(driver);
    nlines = Vect_get_num_lines(Map);
    for (line = 1; line <= nlines; line++) {
        const struct field_rec *f;

        if (!Vect_line_alive(Map, line) ||
            Vect_read_line(Map, Points, Cats, line) != GV_CENTROID)
            continue;
        if (!Vect_cat_get(Cats, 1, &cat) || cat <= 0)
            continue;
        f = &info[cat];
        snprintf(buf, sizeof(buf), "insert into %s values (%d, %d, %d, %.3f)",
                 Fi->table, cat, f->key < 0, f->key < 0 ? -f->key : 0, f->area);
        db_set_string(&sql, buf);
        if (db_execute_immediate(driver, &sql) != DB_OK)
            G_fatal_error(_("Unable to insert new record: %s"),
                          db_get_string(&sql));
    }
    db_commit_transaction(driver);
    db_close_database_shutdown_driver(driver);
    db_free_string(&sql);
    Vect_destroy_line_struct(Points);
    Vect_destroy_cats_struct(Cats);
}

void vectorize_fields(int *ids, int rows, int cols,
                      const struct vect_params *vp)
{
    const size_t n = (size_t)rows * cols;
    struct Cell_head win, fine;
    struct Map_info Map;
    struct line_pnts *Points;
    struct line_cats *Cats;
    struct field_rec *info;
    int *comp, *ckey, *catof, *K, nc, ncat, c, line, nlines;
    int removed_centroids = 0, nbg = 0;
    size_t i;
    double cellarea, removed_area = 0.0;

    G_get_window(&win);
    fine = win;
    fine.rows = rows;
    fine.cols = cols;
    fine.ns_res = win.ns_res / vp->scale;
    fine.ew_res = win.ew_res / vp->scale;
    cellarea = fine.ew_res * fine.ns_res;

    /* Rough shares of the vectorization time per step. */
    progress_begin(PG_VECTOR, 1);
    G_message(_("Filtering fields..."));
    comp = G_malloc(n * sizeof(int));
    nc = label_regions(ids, rows, cols, comp, &ckey);
    progress_part(0.1, 1);
    catof = filter_fields(comp, rows, cols, nc, ckey, vp, &info, &ncat);
    G_free(ckey);

    /* Per-cell category, in place of the components. */
    K = comp;
    for (i = 0; i < n; i++)
        K[i] = catof[K[i]];
    G_free(catof);
    for (c = 1; c <= ncat; c++)
        nbg += info[c].key < 0;
    if (nbg)
        G_message(_("%d fields and %d background areas"), ncat - nbg, nbg);
    else
        G_message(_("%d fields"), ncat);
    if (ncat == 0)
        G_warning(_("No field left: the output vector map will be empty"));

    if (vp->raster_out)
        write_field_raster(vp->raster_out, K, rows, cols, &fine, vp->source);

    progress_part(0.3, 1);
    find_poles(K, ids, rows, cols, info);
    G_free(ids);

    if (Vect_open_new(&Map, vp->vector_out, WITHOUT_Z) < 0)
        G_fatal_error(_("Unable to create vector map <%s>"), vp->vector_out);
    Vect_hist_command(&Map);

    G_message(_("Tracing boundaries..."));
    write_boundaries(&Map, K, rows, cols, &fine, vp->simplify);

    Points = Vect_new_line_struct();
    Cats = Vect_new_cats_struct();
    for (c = 1; c <= ncat; c++) {
        int cell = info[c].best_cell;

        Vect_reset_line(Points);
        Vect_reset_cats(Cats);
        Vect_append_point(Points, fine.west + (cell % cols + 0.5) * fine.ew_res,
                          fine.north - (cell / cols + 0.5) * fine.ns_res, 0.0);
        Vect_cat_set(Cats, 1, c);
        Vect_write_line(&Map, GV_CENTROID, Points, Cats);
    }
    G_free(K);

    /* Topology cleaning: simplification may create crossings, pseudo
     * nodes and slivers. */
    progress_part(0.8, 1);
    G_message(_("Cleaning topology..."));
    Vect_build_partial(&Map, GV_BUILD_BASE);
    Vect_break_lines(&Map, GV_BOUNDARY, NULL);
    Vect_remove_duplicates(&Map, GV_BOUNDARY, NULL);
    Vect_merge_lines(&Map, GV_BOUNDARY, NULL, NULL);
    Vect_build_partial(&Map, GV_BUILD_NONE);
    Vect_build_partial(&Map, GV_BUILD_CENTROIDS);

    nlines = Vect_get_num_lines(&Map);
    for (line = 1; line <= nlines; line++) {
        if (!Vect_line_alive(&Map, line) ||
            Vect_get_line_type(&Map, line) != GV_CENTROID)
            continue;
        if (Vect_get_centroid_area(&Map, line) <= 0) {
            Vect_delete_line(&Map, line);
            removed_centroids++;
        }
    }
    if (removed_centroids)
        G_verbose_message(_("%d centroids fell outside or duplicated an area "
                            "after simplification and were removed"),
                          removed_centroids);
    Vect_remove_small_areas(&Map, cellarea, NULL, &removed_area);
    Vect_build_partial(&Map, GV_BUILD_NONE);
    Vect_build(&Map);

    progress_part(0.95, 1);
    write_attributes(&Map, info);
    Vect_close(&Map);
    progress_end();

    Vect_destroy_line_struct(Points);
    Vect_destroy_cats_struct(Cats);
    G_free(info);
}
