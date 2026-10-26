/****************************************************************************
 *
 * MODULE:       i.delineate.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Imagery group input, normalization to 8-bit RGB, data and
 *               land cover masks, and tile extraction with the bicubic
 *               upsampling of Delineate-Anything's data loader.
 *
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 *****************************************************************************/

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/imagery.h>
#include <grass/raster.h>

#include "image.h"
#include "progress.h"

#define NBINS 65536

static void resolve_group(const char *group, const char *subgroup,
                          struct Ref *ref)
{
    char name[GNAME_MAX], mapset[GMAPSET_MAX];
    const char *found;

    if (!G_name_is_fully_qualified(group, name, mapset)) {
        G_strlcpy(name, group, sizeof(name));
        mapset[0] = '\0';
    }
    found = G_find_file2("group", name, mapset);
    if (!found)
        G_fatal_error(_("Imagery group <%s> not found"), group);

    I_init_group_ref(ref);
    if (subgroup) {
        if (!I_find_subgroup2(name, subgroup, found))
            G_fatal_error(_("Subgroup <%s> not found in group <%s>"), subgroup,
                          group);
        if (!I_get_subgroup_ref2(name, subgroup, found, ref))
            G_fatal_error(_("Unable to read subgroup <%s> of group <%s>"),
                          subgroup, group);
    }
    else if (!I_get_group_ref2(name, found, ref))
        G_fatal_error(_("Unable to read imagery group <%s>"), group);

    if (ref->nfiles != 3)
        G_fatal_error(_("%s <%s> contains %d raster maps; exactly 3 (used "
                        "as R, G, B in group order) are required. Use "
                        "i.group to build a 3-band group or subgroup."),
                      subgroup ? _("Subgroup") : _("Group"),
                      subgroup ? subgroup : group, ref->nfiles);
}

/* Value at percentile pct (0..100) of a histogram over [min, max], with
 * linear interpolation inside the bin. */
static double hist_percentile(const long *hist, long total, double min,
                              double max, double pct)
{
    double target = pct / 100.0 * (total - 1), width = (max - min) / NBINS;
    long cum = 0;
    int b;

    for (b = 0; b < NBINS; b++) {
        if (cum + hist[b] > target) {
            double frac = hist[b] ? (target - cum) / hist[b] : 0.0;

            return min + (b + frac) * width;
        }
        cum += hist[b];
    }
    return max;
}

void image_read_group(const char *group, const char *subgroup, double lo,
                      double hi, struct del_image *img)
{
    struct Ref ref;
    int fd[3], b, row, col, nrows, ncols, bytes = 1;
    DCELL *buf[3];
    double min[3], max[3], vlo[3], vhi[3];
    long *hist, npos[3] = {0, 0, 0}, nvalid = 0;
    size_t n;

    resolve_group(group, subgroup, &ref);

    nrows = Rast_window_rows();
    ncols = Rast_window_cols();
    n = (size_t)nrows * ncols;
    img->rows = nrows;
    img->cols = ncols;
    progress_begin(PG_READ, n / 1e6);
    img->rgb = G_calloc(n * 3, 1);
    img->clip = G_calloc(n, 1);
    img->filter = NULL;
    img->bg = NULL;

    for (b = 0; b < 3; b++) {
        struct Range range;
        CELL rmin, rmax;

        snprintf(img->names[b], sizeof(img->names[b]), "%s@%s",
                 ref.file[b].name, ref.file[b].mapset);
        fd[b] = Rast_open_old(ref.file[b].name, ref.file[b].mapset);
        buf[b] = Rast_allocate_d_buf();
        min[b] = INFINITY;
        max[b] = -INFINITY;
        /* 8-bit data is used as is, as GDAL Byte bands are by
         * Delineate-Anything. */
        if (Rast_map_type(ref.file[b].name, ref.file[b].mapset) != CELL_TYPE ||
            Rast_read_range(ref.file[b].name, ref.file[b].mapset, &range) < 0)
            bytes = 0;
        else {
            Rast_get_range_min_max(&range, &rmin, &rmax);
            if (Rast_is_c_null_value(&rmin) || rmin < 0 || rmax > 255)
                bytes = 0;
        }
    }
    G_message(_("Input bands (R, G, B): <%s>, <%s>, <%s>"), img->names[0],
              img->names[1], img->names[2]);

    /* Pass 1: valid data and range of the positive values per band. */
    G_message(_("Reading imagery group <%s>..."), group);
    for (row = 0; row < nrows; row++) {
        progress_part(row, 3.0 * nrows);
        for (b = 0; b < 3; b++)
            Rast_get_d_row(fd[b], buf[b], row);
        for (col = 0; col < ncols; col++) {
            size_t i = (size_t)row * ncols + col;
            int nonzero = 0;

            for (b = 0; b < 3; b++) {
                double v = buf[b][col];

                if (Rast_is_d_null_value(&buf[b][col]))
                    break;
                nonzero |= v != 0.0;
                if (v > 0) {
                    npos[b]++;
                    if (v < min[b])
                        min[b] = v;
                    if (v > max[b])
                        max[b] = v;
                }
            }
            if (b == 3 && nonzero) {
                img->clip[i] = 1;
                nvalid++;
            }
        }
    }
    if (nvalid == 0)
        G_fatal_error(_("No cell has valid data in all three bands in the "
                        "current region"));

    if (bytes) {
        G_verbose_message(_("8-bit input: no stretch"));
        for (b = 0; b < 3; b++) {
            vlo[b] = 0;
            vhi[b] = 255;
        }
    }
    else {
        /* Pass 2: per-band histograms of the positive values. */
        hist = G_calloc(3L * NBINS, sizeof(long));
        for (row = 0; row < nrows; row++) {
            progress_part(nrows + row, 3.0 * nrows);
            for (b = 0; b < 3; b++)
                Rast_get_d_row(fd[b], buf[b], row);
            for (col = 0; col < ncols; col++)
                for (b = 0; b < 3; b++) {
                    double v = buf[b][col], r;
                    int k;

                    if (Rast_is_d_null_value(&buf[b][col]) || !(v > 0))
                        continue;
                    r = max[b] > min[b] ? (v - min[b]) / (max[b] - min[b])
                                        : 0.0;
                    k = (int)(r * NBINS);
                    hist[(long)b * NBINS + (k >= NBINS ? NBINS - 1 : k)]++;
                }
        }
        for (b = 0; b < 3; b++) {
            if (npos[b] == 0)
                G_fatal_error(_("Band <%s> has no positive value in the "
                                "current region"),
                              img->names[b]);
            vlo[b] = hist_percentile(hist + (long)b * NBINS, npos[b], min[b],
                                     max[b], lo);
            vhi[b] = hist_percentile(hist + (long)b * NBINS, npos[b], min[b],
                                     max[b], hi);
            if (!(vhi[b] > vlo[b])) {
                vlo[b] = min[b];
                vhi[b] = max[b];
            }
            if (!(vhi[b] > vlo[b]))
                G_warning(_("Band <%s> is constant in the current region"),
                          img->names[b]);
            G_verbose_message(_("Band <%s>: stretch %g .. %g (positive "
                                "values %g .. %g)"),
                              img->names[b], vlo[b], vhi[b], min[b], max[b]);
        }
        G_free(hist);
    }

    /* Pass 3: 8-bit conversion, truncating as numpy's astype("uint8"). */
    for (row = 0; row < nrows; row++) {
        progress_part(2.0 * nrows + row, 3.0 * nrows);
        for (b = 0; b < 3; b++)
            Rast_get_d_row(fd[b], buf[b], row);
        for (col = 0; col < ncols; col++) {
            size_t i = (size_t)row * ncols + col;

            if (!img->clip[i])
                continue;
            for (b = 0; b < 3; b++) {
                double v = vhi[b] > vlo[b] ? 255.0 * (buf[b][col] - vlo[b]) /
                                                 (vhi[b] - vlo[b])
                                           : 0.0;

                img->rgb[3 * i + b] = v <= 0.0     ? 0
                                      : v >= 255.0 ? 255
                                                   : (unsigned char)v;
            }
        }
    }

    for (b = 0; b < 3; b++) {
        Rast_close(fd[b]);
        G_free(buf[b]);
    }
    I_free_group_ref(&ref);
    progress_end();
}

static int in_list(int v, const int *list, int n)
{
    int i;

    for (i = 0; i < n; i++)
        if (list[i] == v)
            return 1;
    return 0;
}

void image_read_mask(const struct del_mask *mask, struct del_image *img)
{
    const int rows = img->rows, cols = img->cols;
    size_t n = (size_t)rows * cols;
    unsigned char *valid = img->clip;
    int r, c;

    /* Filter mask: the valid data eroded by a 5 x 5 square, cells outside
     * the region counting as no data. */
    img->filter = G_malloc(n);
    {
        unsigned char *tmp = G_malloc(n);

#pragma omp parallel for private(c)
        for (r = 0; r < rows; r++)
            for (c = 0; c < cols; c++) {
                int v = c >= 2 && c < cols - 2, k;

                for (k = -2; v && k <= 2; k++)
                    v = valid[(size_t)r * cols + c + k];
                tmp[(size_t)r * cols + c] = (unsigned char)v;
            }
#pragma omp parallel for private(c)
        for (r = 0; r < rows; r++)
            for (c = 0; c < cols; c++) {
                int v = r >= 2 && r < rows - 2, k;

                for (k = -2; v && k <= 2; k++)
                    v = tmp[(size_t)(r + k) * cols + c];
                img->filter[(size_t)r * cols + c] = (unsigned char)v;
            }
        G_free(tmp);
    }

    if (!mask->name)
        return;

    {
        int fd = Rast_open_old(mask->name, "");
        CELL *buf = Rast_allocate_c_buf();

        if (mask->nbg)
            img->bg = G_calloc(n, sizeof(int));
        G_message(_("Reading land cover mask <%s>..."), mask->name);
        for (r = 0; r < rows; r++) {
            Rast_get_c_row(fd, buf, r);
            for (c = 0; c < cols; c++) {
                size_t i = (size_t)r * cols + c;
                int v;

                if (Rast_is_c_null_value(&buf[c]))
                    continue;
                v = buf[c];
                if (in_list(v, mask->clip, mask->nclip))
                    img->clip[i] = 0;
                if (in_list(v, mask->filter, mask->nfilter))
                    img->filter[i] = 0;
                if (img->bg && in_list(v, mask->bg, mask->nbg))
                    img->bg[i] = v;
            }
        }
        Rast_close(fd);
        G_free(buf);
    }
}

/* OpenCV's bicubic kernel (A = -0.75) and its 8-bit fixed point
 * resampling: 11-bit coefficients, horizontal pass to integers, vertical
 * pass rounded and saturated, replicated borders. */

#define COEF_BITS 11
#define COEF_SCALE (1 << COEF_BITS)

static void cubic_coeffs(float x, int *k)
{
    const float A = -0.75f;
    float c[4];
    int i;

    c[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    c[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    c[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    c[3] = 1.f - c[0] - c[1] - c[2];
    for (i = 0; i < 4; i++)
        k[i] = (int)lrintf(c[i] * COEF_SCALE);
}

/* Source taps and fixed-point weights of output pixel d when resizing in
 * pixels to out pixels. */
static void cubic_taps(int d, int in, int out, int *idx, int *k)
{
    double scale = (double)in / out;
    float f = (float)((d + 0.5) * scale - 0.5);
    int s = (int)floorf(f), i;

    f -= s;
    cubic_coeffs(f, k);
    for (i = 0; i < 4; i++) {
        int j = s - 1 + i;

        idx[i] = j < 0 ? 0 : j >= in ? in - 1 : j;
    }
}

int image_tile_valid(const struct del_image *img, int x0, int y0, int t)
{
    int r, c;

    for (r = y0 > 0 ? y0 : 0; r < y0 + t && r < img->rows; r++)
        for (c = x0 > 0 ? x0 : 0; c < x0 + t && c < img->cols; c++)
            if (img->clip[(size_t)r * img->cols + c])
                return 1;
    return 0;
}

int image_tile(const struct del_image *img, int x0, int y0, int t, int size,
               float *planes, unsigned char *clip, unsigned char *filter)
{
    const int rows = img->rows, cols = img->cols;
    unsigned char *rgb, *cl, *fl;
    int x, y, ch, any = 0;
    const size_t plane = (size_t)size * size;

    /* The tile at the region resolution; outside the region it is black
     * no data. */
    rgb = G_calloc((size_t)t * t * 3, 1);
    cl = G_calloc((size_t)t * t, 1);
    fl = G_calloc((size_t)t * t, 1);
    for (y = 0; y < t; y++) {
        int r = y0 + y;

        if (r < 0 || r >= rows)
            continue;
        for (x = 0; x < t; x++) {
            int c = x0 + x;
            size_t i;

            if (c < 0 || c >= cols)
                continue;
            i = (size_t)r * cols + c;
            memcpy(rgb + ((size_t)y * t + x) * 3, img->rgb + 3 * i, 3);
            cl[(size_t)y * t + x] = img->clip[i];
            fl[(size_t)y * t + x] = img->filter[i];
            any |= img->clip[i];
        }
    }
    if (!any) {
        G_free(rgb);
        G_free(cl);
        G_free(fl);
        return 0;
    }

    if (size == t) {
        for (ch = 0; ch < 3; ch++)
            for (y = 0; y < t; y++)
                for (x = 0; x < t; x++)
                    planes[ch * plane + (size_t)y * t + x] =
                        rgb[((size_t)y * t + x) * 3 + ch] / 255.0f;
        memcpy(clip, cl, plane);
        memcpy(filter, fl, plane);
    }
    else {
        int *hx = G_malloc(sizeof(int) * 4 * size);
        int *kx = G_malloc(sizeof(int) * 4 * size);
        int *hbuf = G_malloc(sizeof(int) * (size_t)t * size * 3);

        for (x = 0; x < size; x++)
            cubic_taps(x, t, size, hx + 4 * x, kx + 4 * x);
        /* Horizontal pass over every source row. */
        for (y = 0; y < t; y++)
            for (x = 0; x < size; x++)
                for (ch = 0; ch < 3; ch++) {
                    int s = 0, i;

                    for (i = 0; i < 4; i++)
                        s += rgb[((size_t)y * t + hx[4 * x + i]) * 3 + ch] *
                             kx[4 * x + i];
                    hbuf[((size_t)y * size + x) * 3 + ch] = s;
                }
        /* Vertical pass. */
        for (y = 0; y < size; y++) {
            int vy[4], ky[4];

            cubic_taps(y, t, size, vy, ky);
            for (x = 0; x < size; x++)
                for (ch = 0; ch < 3; ch++) {
                    int64_t s = 1 << (2 * COEF_BITS - 1);
                    int i, v;

                    for (i = 0; i < 4; i++)
                        s +=
                            (int64_t)hbuf[((size_t)vy[i] * size + x) * 3 + ch] *
                            ky[i];
                    v = (int)(s >> (2 * COEF_BITS));
                    v = v < 0 ? 0 : v > 255 ? 255 : v;
                    planes[ch * plane + (size_t)y * size + x] = v / 255.0f;
                }
        }
        /* Masks: nearest neighbour, floor((d + 0.5) * t / size). */
        for (y = 0; y < size; y++) {
            int sy = (int)((y + 0.5) * t / size);

            for (x = 0; x < size; x++) {
                int sx = (int)((x + 0.5) * t / size);

                clip[(size_t)y * size + x] = cl[(size_t)sy * t + sx];
                filter[(size_t)y * size + x] = fl[(size_t)sy * t + sx];
            }
        }
        G_free(hx);
        G_free(kx);
        G_free(hbuf);
    }
    G_free(rgb);
    G_free(cl);
    G_free(fl);
    return 1;
}

void image_free(struct del_image *img)
{
    G_free(img->rgb);
    G_free(img->clip);
    G_free(img->filter);
    G_free(img->bg);
    img->rgb = img->clip = img->filter = NULL;
    img->bg = NULL;
}
