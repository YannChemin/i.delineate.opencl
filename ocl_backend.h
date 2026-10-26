/* SPDX-License-Identifier: Unlicense */

#ifndef I_DELINEATE_OPENCL_OCL_BACKEND_H
#define I_DELINEATE_OPENCL_OCL_BACKEND_H

/* Target the OpenCL 1.2 API, the common denominator of PoCL, Mesa
 * rusticl/Clover and the discrete GPU vendor ICDs. */
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>

#include <stddef.h>

struct ocl_backend {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    char device_name[256];
    char platform_name[256];
    cl_ulong max_alloc;
    cl_ulong global_mem;
    cl_ulong local_mem;
    cl_uint compute_units;
    int is_gpu;
    int unified;       /* CL_DEVICE_HOST_UNIFIED_MEMORY: memory shared with
                        * the host (iGPU, APU, CPU). */
    int has_fp16;      /* cl_khr_fp16. */
    char version[128]; /* OpenCL C version. */
    char extensions[4096];
    double link_gbs; /* Narrowest PCIe link to the device in GB/s, 0
                      * when unknown. */
    int profile;     /* Per-kernel timing (I_DELINEATE_OPENCL_PROFILE set). */
};

/* Called once the device is chosen and described, before the kernels are
 * built: returns the extra build options. */
typedef const char *(*ocl_opts_fn)(struct ocl_backend *ocl, void *data);

/* Select a device and build the embedded kernels.
 *
 * device_opt is "auto" (first GPU, else first CPU device), "gpu" or
 * "cpu". platform_opt, when not NULL, restricts the search to platforms
 * whose name contains that substring (case-insensitive), e.g. "rusticl"
 * or "Portable". Fails with G_fatal_error() if no device matches: this
 * module has no non-OpenCL fallback. opts, when not NULL, gives the extra
 * compiler options once the device is known. */
void ocl_init(struct ocl_backend *ocl, const char *device_opt,
              const char *platform_opt, ocl_opts_fn opts, void *data);

/* A second in-order queue on the same device, for transfers. */
cl_command_queue ocl_queue(struct ocl_backend *ocl);

void ocl_free(struct ocl_backend *ocl);

/* Human-readable name of an OpenCL error code. */
const char *ocl_errstr(cl_int err);

/* Fatal error helper: aborts with a message naming the failed call. */
void ocl_check(cl_int err, const char *what);

/* Allocate a device buffer of n bytes, fatal on failure. */
cl_mem ocl_alloc(struct ocl_backend *ocl, size_t n, const char *what);

/* Allocate and fill a device buffer from host memory. */
cl_mem ocl_upload(struct ocl_backend *ocl, const void *src, size_t n,
                  const char *what);

void ocl_write(struct ocl_backend *ocl, cl_mem dst, size_t off, const void *src,
               size_t n);
void ocl_read(struct ocl_backend *ocl, cl_mem src, size_t off, void *dst,
              size_t n);

cl_kernel ocl_kernel(struct ocl_backend *ocl, const char *name);

void ocl_release(cl_mem *m);

#endif /* I_DELINEATE_OPENCL_OCL_BACKEND_H */
