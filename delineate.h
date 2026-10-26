/* SPDX-License-Identifier: AGPL-3.0-only */

#ifndef I_DELINEATE_OPENCL_DELINEATE_H
#define I_DELINEATE_OPENCL_DELINEATE_H

#include "image.h"
#include "yolo_model.h"

/* Parameters of Delineate-Anything's "main" method, named after its
 * configuration file (conf_sample.yaml). */
struct del_params {
    int scale;        /* Output cells per region cell along each axis. */
    double tile_step; /* Tile step as a fraction of the tile size. */
    float conf;       /* minimal_confidence */
    float nms_iou;    /* Ultralytics NMS IoU threshold. */
    int max_det;
    int batch;
    /* delineation_config */
    int pixel_area_threshold;
    double remaining_area_threshold;
    double compose_merge_iou;
    double merge_iou;
    double merge_relative_area_threshold;
    int merge_asymetric_pixel_area_threshold;
    double merge_asymetric_relative_area_threshold;
    int merging_edge_width;
    double merge_edge_iou;
    int merge_edge_pixels;
    const char *dump_dir; /* Developer dumps, or NULL. */
};

/* Number of tiles that the region yields (with data or not), for the
 * progress plan. */
long delineate_count_tiles(const struct del_image *img,
                           const struct del_params *p);

/* Run the network over the tiles of the region, compose the fields of
 * each tile, merge them across tiles, and return the field identity
 * raster of (rows * scale) x (cols * scale) cells: 0 or 1 where there is
 * no field, a field identity >= 2, or minus the land cover class of a
 * background cell. */
int *delineate_run(struct yolo_model *m, const struct del_image *img,
                   const struct del_params *p);

#endif /* I_DELINEATE_OPENCL_DELINEATE_H */
