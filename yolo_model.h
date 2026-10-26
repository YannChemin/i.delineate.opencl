/* SPDX-License-Identifier: AGPL-3.0-only */

#ifndef I_DELINEATE_OPENCL_YOLO_MODEL_H
#define I_DELINEATE_OPENCL_YOLO_MODEL_H

#include "ocl_backend.h"
#include "pth_loader.h"

/* Side of the square network input, in pixels: every Delineate-Anything
 * model was trained on 512 x 512 tiles. */
#define YOLO_IMG 512

/* Blocking of the conv_gemm kernel: work-groups of WG x WG work-items,
 * each computing RB x RB outputs, stepping through the reduction by
 * TILE_K. The kernels are built with these values (YOLO_BUILD_OPTS), so
 * host and device always agree. */
#define YOLO_WG 16
#define YOLO_RB 4
#define YOLO_TILE_K 16
#define YOLO_TILE (YOLO_WG * YOLO_RB)
#define YOLO_STR2(x) #x
#define YOLO_STR(x) YOLO_STR2(x)
/* clang-format off */
#define YOLO_BUILD_OPTS                                                        \
    "-DWG=" YOLO_STR(YOLO_WG)                                                  \
    " -DRB=" YOLO_STR(YOLO_RB)                                                 \
    " -DTILE_K=" YOLO_STR(YOLO_TILE_K)
/* clang-format on */

/* Device profiles: how the network uses the device it runs on.
 *   cpu   CPU device (PoCL): convolutions as host BLAS matrix products
 *         on mapped buffers, one batch at a time.
 *   igpu  GPU sharing memory with the host: zero-copy mapped input and
 *         outputs, two batches in flight.
 *   dgpu  discrete GPU: pinned staging buffers, two batches in flight.
 *   egpu  GPU behind a slow link (Thunderbolt, PoCL-Remote): as dgpu,
 *         with transfers on their own queue overlapping the compute. */
enum yolo_profile { YP_CPU, YP_IGPU, YP_DGPU, YP_EGPU };

/* Device setup chosen before the kernels are built (see
 * yolo_device_opts()). */
struct yolo_setup {
    const char *profile_opt;   /* auto, cpu, igpu, dgpu or egpu. */
    const char *precision_opt; /* auto, fp32 or fp16. */
    enum yolo_profile profile;
    int half; /* Activations stored as float16. */
    char opts[256];
};

/* ocl_init() callback: choose the profile and precision for the device
 * (fatal when a forced one cannot run there) and return the kernel build
 * options. data is a struct yolo_setup. */
const char *yolo_device_opts(struct ocl_backend *ocl, void *data);

extern const char *const yolo_profile_names[];

/* A slice of channels of a device tensor: element offset of the first
 * channel of image 0, elements from one image to the next, and the shape
 * of one image. */
struct yolo_view {
    cl_mem buf;
    long off, bs;
    int c, h, w;
};

/* A convolution with batch normalization folded in: k x k, stride s, one
 * group (GEMM kernel) or depthwise 3 x 3. A 2 x 2 stride 2 transposed
 * convolution is stored as a 1 x 1 convolution with 4 * cout outputs
 * written through a 2x pixel shuffle. */
struct yolo_conv {
    int cin, cout, k, s, pad;
    int dw, silu, shuffle;
    int mpad;
    cl_mem w, b;
    float *hw, *hb; /* cpu profile: (rows, cin * k * k) weights, bias. */
};

struct yolo_chain {
    int n;
    struct yolo_conv *v;
};

struct yolo_bottleneck {
    struct yolo_conv cv1, cv2;
    int add;
};

/* C3k2/C2f items: a Bottleneck, or a C3k (a CSP block of bottlenecks). */
struct yolo_item {
    int c3k;
    struct yolo_bottleneck b;
    struct yolo_conv cv1, cv2, cv3;
    int nb;
    struct yolo_bottleneck *m;
};

struct yolo_psa {
    struct yolo_conv qkv, pe, proj, ffn0, ffn1;
    int heads, kd, hd, add;
    float scale;
};

enum yolo_kind {
    YL_CONV,
    YL_C2F, /* C2f and C3k2. */
    YL_SPPF,
    YL_C2PSA,
    YL_UPSAMPLE,
    YL_CONCAT,
    YL_SEGMENT
};

struct yolo_layer {
    enum yolo_kind kind;
    int nf, f[4]; /* Input layers; -1 is the previous one. */
    struct yolo_conv cv1, cv2;
    int c, n;                /* Hidden channels and item count. */
    struct yolo_item *items; /* YL_C2F. */
    struct yolo_psa *psa;    /* YL_C2PSA. */
    int pool;                /* YL_SPPF window. */
    struct yolo_view out;
};

/* The segmentation head: per detection level the box distribution
 * (4 x reg_max), class and mask coefficient branches, and the mask
 * prototype network on the finest level. */
struct yolo_head {
    int nl, nc, nm, reg_max;
    struct yolo_chain box[4], cls[4], mc[4];
    struct yolo_conv p_cv1, p_up, p_cv2, p_cv3;
    struct yolo_view lvl[4]; /* (4 reg_max + nc + nm) channels each. */
    struct yolo_view proto;  /* nm channels at twice the finest level. */
    int stride[4];
};

/* A batch in flight: device input and head outputs, and their host
 * side. */
struct yolo_slot {
    cl_mem input, lvl[4], proto;
    cl_mem staging; /* dgpu/egpu: pinned host buffer of the input. */
    void *stage;    /* Its mapping, or the mapped device input. */
    float *in;      /* Input handed to the caller. */
    float *out_lvl[4], *out_proto; /* Head outputs as float32. */
    void *raw_lvl[4], *raw_proto;  /* Read back or mapped outputs. */
    int own_out;                   /* out_* are host arrays (not mappings). */
    int nb;
    cl_event done;
};

/* Per-kernel totals of the profiling switch. */
struct yolo_prof {
    const char *name;
    long calls;
    double sec, flop, bytes;
};

struct yolo_model {
    struct ocl_backend *ocl;
    struct yolo_setup setup;
    int esize; /* Bytes per activation. */
    int nlayers, batch;
    struct yolo_layer *layers;
    struct yolo_head head;
    cl_mem input; /* (batch, 3, YOLO_IMG, YOLO_IMG), RGB in 0..1, of the
                   * slot being run. */
    cl_mem scratch;
    long scratch_top, scratch_peak;
    int planning;
    double gflop; /* Per image, for the log. */
    cl_kernel k_gemm, k_dw, k_pool, k_up, k_copy, k_attn;
    cl_command_queue xfer; /* Transfer queue (egpu), else the queue. */
    int nslots;
    struct yolo_slot slot[2];
    /* cpu profile: buffers mapped for the current forward pass. */
#define YOLO_MAX_MAPPED 64
    int nmapped;
    cl_mem mapped[YOLO_MAX_MAPPED];
    float *mapped_ptr[YOLO_MAX_MAPPED];
    int nprof;
    struct yolo_prof prof[16];
};

/* Build the network described by the module tree of a Delineate-Anything
 * (Ultralytics YOLO segmentation) checkpoint, upload its weights and
 * allocate the activations of batch images. Fatal if the checkpoint
 * holds anything else. */
void yolo_load(struct yolo_model *m, struct ocl_backend *ocl,
               const struct yolo_setup *setup, const struct pth_file *pf,
               int batch);

/* Batches go through slots: the caller fills yolo_input() with nb images
 * of 3 x YOLO_IMG x YOLO_IMG floats (RGB planes scaled to 0..1), submits
 * them, may fill and submit the other slot, then waits for a slot before
 * reading its detections and releases it. */
float *yolo_input(struct yolo_model *m, int s);
void yolo_submit(struct yolo_model *m, int s, int nb);
void yolo_wait(struct yolo_model *m, int s);
void yolo_release(struct yolo_model *m, int s);

/* Print the profiling totals (I_DELINEATE_OPENCL_PROFILE). */
void yolo_report(const struct yolo_model *m);

/* One detected field: its box in tile pixels and its mask over the
 * window (wx0, wy0, ww, wh) of the tile, the box widened by 4 pixels as
 * the reference post-processing crops it. */
struct yolo_det {
    float x1, y1, x2, y2, score;
    int wx0, wy0, ww, wh;
    unsigned char *mask;
};

struct yolo_dets {
    int n, cap;
    struct yolo_det *v;
};

/* Detections of image b of the batch waited for in slot s: confidence above
 * conf, non-maximum suppression at IoU iou, at most max_det boxes, masks from
 * the prototypes at full tile resolution, then opened and closed with a
 * 3 x 3 square (Ultralytics predict(retina_masks=True) followed by the
 * Delineate-Anything mask morphology). Detections with an empty mask are
 * dropped. */
void yolo_detect(const struct yolo_model *m, int s, int b, float conf,
                 float iou, int max_det, struct yolo_dets *out);

void yolo_dets_free(struct yolo_dets *d);

/* Write the outputs of every layer of image 0 of the batch waited for in
 * slot 0 into dir
 * (developer aid, I_DELINEATE_OPENCL_DUMP). */
void yolo_dump(struct yolo_model *m, const char *dir);

void yolo_free(struct yolo_model *m);

#endif /* I_DELINEATE_OPENCL_YOLO_MODEL_H */
