/* SPDX-License-Identifier: AGPL-3.0-only */

#ifndef I_DELINEATE_OPENCL_VECTORIZE_H
#define I_DELINEATE_OPENCL_VECTORIZE_H

struct vect_params {
    int scale;          /* Output cells per region cell along each axis. */
    double min_area;    /* Square meters; smaller fields are dropped. */
    double min_hole;    /* Square meters; smaller holes are filled. */
    double bg_min_area; /* The same for background (land cover) fields. */
    double bg_min_hole;
    double simplify; /* Douglas-Peucker tolerance in map units, 0 = off. */
    const char *vector_out;
    const char *raster_out; /* Optional field raster, may be NULL. */
    const char *source;     /* Input description for the history. */
};

/* Turn the field identity raster of delineate_run() (rows x cols cells at
 * the output resolution, freed here) into a topological vector area map,
 * one area and attribute row per field. */
void vectorize_fields(int *ids, int rows, int cols,
                      const struct vect_params *vp);

#endif /* I_DELINEATE_OPENCL_VECTORIZE_H */
