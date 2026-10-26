/* SPDX-License-Identifier: Unlicense */

/* Whole-run progress: one G_percent() bar from the first band read to the
 * last vector written, which MetaMonitor's GRASS hook picks up
 * (https://github.com/YannChemin/MetaMonitor). */

#ifndef I_DELINEATE_OPENCL_PROGRESS_H
#define I_DELINEATE_OPENCL_PROGRESS_H

#include <stddef.h>

enum progress_kind {
    PG_READ,   /* Input bands, in millions of cells. */
    PG_LOAD,   /* Checkpoint tensors, in GiB. */
    PG_BUILD,  /* OpenCL program build. */
    PG_INFER,  /* Network and per-tile merging, per tile. */
    PG_MERGE,  /* Field identity resolution over the region. */
    PG_VECTOR, /* Field filtering and vectorization, per run. */
    PG_NKINDS
};

/* Add units of work of one kind to the run's plan. */
void progress_plan(enum progress_kind k, double units);

/* Start units of work of one kind (planned or not), advance through them
 * with progress_part(), and close them with progress_end(). */
void progress_begin(enum progress_kind k, double units);
void progress_part(double done, double total);
void progress_end(void);

/* Bytes read from the checkpoint, advancing an open PG_LOAD stage. */
void progress_bytes(size_t n);

/* Report 100 %. */
void progress_finish(void);

#endif /* I_DELINEATE_OPENCL_PROGRESS_H */
