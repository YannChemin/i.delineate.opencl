/* SPDX-License-Identifier: AGPL-3.0-only */

#ifndef I_DELINEATE_OPENCL_IMAGE_H
#define I_DELINEATE_OPENCL_IMAGE_H

#include <grass/gis.h>

/* The current region as the 8-bit RGB image Delineate-Anything works on,
 * with its masks, all at the region resolution. */
struct del_image {
    int rows, cols;
    unsigned char *rgb;    /* rows * cols * 3, band order of the group. */
    unsigned char *clip;   /* 1 where fields may lie: valid data outside
                            * the clip classes. */
    unsigned char *filter; /* 1 where fields count as valid: data at least
                            * 2 cells inside the valid area, outside the
                            * filter classes. */
    int *bg;               /* Land cover class where it is a background
                            * class, else 0; NULL without background. */
    char names[3][GNAME_MAX + GMAPSET_MAX + 1];
};

/* Land cover mask and its class lists (counts may be 0). */
struct del_mask {
    const char *name; /* Raster map, or NULL for no mask. */
    int nclip, nfilter, nbg;
    const int *clip, *filter, *bg;
};

/* Read the three raster maps of an imagery group (or of one of its
 * subgroups) over the current region. Integer maps within 0..255 are
 * used as they are; other maps are stretched linearly to 0..255 between
 * the lo and hi percentiles of their positive values. Cells that are
 * NULL in any band, or 0 in all three, are no data. Fatal unless the
 * (sub)group holds exactly three maps. */
void image_read_group(const char *group, const char *subgroup, double lo,
                      double hi, struct del_image *img);

/* Derive the clip and filter masks from the valid data and the optional
 * land cover mask. */
void image_read_mask(const struct del_mask *mask, struct del_image *img);

/* Whether the tile of t x t cells at (x0, y0) has a cell where fields
 * may lie. */
int image_tile_valid(const struct del_image *img, int x0, int y0, int t);

/* Network input for the tile of t x t cells whose top-left cell is
 * (x0, y0) (cells outside the region are no data): RGB planes of
 * size x size in 0..1 (OpenCV bicubic upsampling when size > t), and the
 * clip and filter masks at size x size (nearest neighbour). Returns 0,
 * filling nothing, when the tile has no cell where fields may lie. */
int image_tile(const struct del_image *img, int x0, int y0, int t, int size,
               float *planes, unsigned char *clip, unsigned char *filter);

void image_free(struct del_image *img);

#endif /* I_DELINEATE_OPENCL_IMAGE_H */
