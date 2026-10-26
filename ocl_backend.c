/****************************************************************************
 *
 * MODULE:       i.delineate.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      OpenCL platform/device selection, program build and
 *               buffer helpers.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "ocl_backend.h"

/* yolo_kernels.cl, turned into a string literal by the Makefile. */
static const char *yolo_kernel_source =
#include "yolo_kernels_cl.h"
    ;

const char *ocl_errstr(cl_int err)
{
    switch (err) {
    case CL_SUCCESS:
        return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
    default:
        return "unknown OpenCL error";
    }
}

void ocl_check(cl_int err, const char *what)
{
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: %s failed: %s (%d)"), what, ocl_errstr(err),
                      err);
}

static int contains_nocase(const char *hay, const char *needle)
{
    size_t n = strlen(needle), i, j;

    for (i = 0; hay[i]; i++) {
        for (j = 0; j < n && hay[i + j]; j++)
            if (tolower((unsigned char)hay[i + j]) !=
                tolower((unsigned char)needle[j]))
                break;
        if (j == n)
            return 1;
    }
    return n == 0;
}

struct candidate {
    cl_platform_id platform;
    cl_device_id device;
};

/* Append the devices of the given type found on platforms matching
 * platform_opt, in platform order. */
static int list_devices(cl_device_type type, const char *platform_opt,
                        struct candidate *out, int n, int max)
{
    cl_platform_id platforms[16];
    cl_device_id devices[8];
    cl_uint np, nd, p, d;
    char pname[256];
    int k;

    if (clGetPlatformIDs(16, platforms, &np) != CL_SUCCESS)
        return n;
    for (p = 0; p < np; p++) {
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(pname), pname,
                          NULL);
        if (platform_opt && !contains_nocase(pname, platform_opt))
            continue;
        if (clGetDeviceIDs(platforms[p], type, 8, devices, &nd) != CL_SUCCESS)
            continue;
        for (d = 0; d < nd && n < max; d++) {
            for (k = 0; k < n; k++)
                if (out[k].device == devices[d])
                    break;
            if (k == n) {
                out[n].platform = platforms[p];
                out[n++].device = devices[d];
            }
        }
    }
    return n;
}

#ifndef CL_DEVICE_PCI_BUS_INFO_KHR
#define CL_DEVICE_PCI_BUS_INFO_KHR 0x410F
#endif

struct pci_bus_info {
    cl_uint domain, bus, device, function;
};

/* Bandwidth in GB/s of the PCIe link of one sysfs device folder, 0 if it
 * has none. */
static double link_gbs(const char *dir)
{
    char path[PATH_MAX], buf[64];
    double gts = 0;
    int width = 0;
    FILE *fp;

    snprintf(path, sizeof(path), "%s/current_link_speed", dir);
    if ((fp = fopen(path, "r"))) {
        if (fgets(buf, sizeof(buf), fp))
            gts = atof(buf);
        fclose(fp);
    }
    snprintf(path, sizeof(path), "%s/current_link_width", dir);
    if ((fp = fopen(path, "r"))) {
        if (fgets(buf, sizeof(buf), fp))
            width = atoi(buf);
        fclose(fp);
    }
    if (gts <= 0 || width <= 0)
        return 0;
    /* 8b/10b encoding up to 5 GT/s, 128b/130b above. */
    return gts * width * (gts > 5.1 ? 128.0 / 130.0 : 0.8) / 8.0;
}

/* Narrowest PCIe link between the root complex and the device, from
 * cl_khr_pci_bus_info and sysfs: a Thunderbolt eGPU sits behind a link
 * of about 4 GB/s even when its own slot reports x16. */
static double device_link_gbs(cl_device_id dev, const char *ext)
{
    struct pci_bus_info pci;
    char path[PATH_MAX], real[PATH_MAX], *p;
    double best = 0;

    if (!strstr(ext, "cl_khr_pci_bus_info") ||
        clGetDeviceInfo(dev, CL_DEVICE_PCI_BUS_INFO_KHR, sizeof(pci), &pci,
                        NULL) != CL_SUCCESS)
        return 0;
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.%x",
             pci.domain, pci.bus, pci.device, pci.function);
    if (!realpath(path, real))
        return 0;
    /* Every PCI address component of the device path is a link. */
    for (p = real + strlen(real); p > real; p--) {
        if (*p == '/') {
            double g;

            if (!strchr(p + 1, ':'))
                break;
            g = link_gbs(real);
            if (g > 0 && (best == 0 || g < best))
                best = g;
            *p = '\0';
        }
    }
    return best;
}

/* Set up context, queue and program on one device. Returns 0 and fills
 * log (if not NULL) when the kernels do not build there. */
static int try_device(struct ocl_backend *ocl, const struct candidate *c,
                      ocl_opts_fn optfn, void *data, char **log)
{
    const char *src = yolo_kernel_source;
    size_t len = strlen(src);
    char opts[1024];
    cl_bool unified = CL_FALSE;
    cl_device_type type;
    cl_int err;

    ocl->platform = c->platform;
    ocl->device = c->device;
    clGetPlatformInfo(c->platform, CL_PLATFORM_NAME, sizeof(ocl->platform_name),
                      ocl->platform_name, NULL);
    clGetDeviceInfo(c->device, CL_DEVICE_NAME, sizeof(ocl->device_name),
                    ocl->device_name, NULL);
    clGetDeviceInfo(c->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
                    &ocl->max_alloc, NULL);
    clGetDeviceInfo(c->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                    &ocl->global_mem, NULL);
    clGetDeviceInfo(c->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
                    &ocl->local_mem, NULL);
    clGetDeviceInfo(c->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                    &ocl->compute_units, NULL);
    clGetDeviceInfo(c->device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified),
                    &unified, NULL);
    clGetDeviceInfo(c->device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    ocl->is_gpu = (type & CL_DEVICE_TYPE_GPU) != 0;
    ocl->unified = unified == CL_TRUE || (type & CL_DEVICE_TYPE_CPU) != 0;
    ocl->extensions[0] = '\0';
    clGetDeviceInfo(c->device, CL_DEVICE_EXTENSIONS, sizeof(ocl->extensions),
                    ocl->extensions, NULL);
    ocl->has_fp16 = strstr(ocl->extensions, "cl_khr_fp16") != NULL;
    ocl->link_gbs = ocl->is_gpu && !ocl->unified
                        ? device_link_gbs(c->device, ocl->extensions)
                        : 0;

    /* The kernels are OpenCL C 1.1; request 1.2 where available. */
    ocl->version[0] = '\0';
    clGetDeviceInfo(c->device, CL_DEVICE_OPENCL_C_VERSION, sizeof(ocl->version),
                    ocl->version, NULL);
    snprintf(opts, sizeof(opts), "%s -cl-mad-enable %s",
             strncmp(ocl->version, "OpenCL C 1.1", 12) == 0 ||
                     strncmp(ocl->version, "OpenCL C 1.0", 12) == 0
                 ? "-cl-std=CL1.1"
                 : "-cl-std=CL1.2",
             optfn ? optfn(ocl, data) : "");

    ocl->context = clCreateContext(NULL, 1, &c->device, NULL, NULL, &err);
    ocl_check(err, "clCreateContext");
    /* Developer aid: I_DELINEATE_OPENCL_PROFILE=1 times every kernel. */
    ocl->profile = getenv("I_DELINEATE_OPENCL_PROFILE") != NULL;
    ocl->queue = clCreateCommandQueue(
        ocl->context, c->device, ocl->profile ? CL_QUEUE_PROFILING_ENABLE : 0,
        &err);
    ocl_check(err, "clCreateCommandQueue");
    ocl->program = clCreateProgramWithSource(ocl->context, 1, &src, &len, &err);
    ocl_check(err, "clCreateProgramWithSource");

    err = clBuildProgram(ocl->program, 1, &c->device, opts, NULL, NULL);
    if (err == CL_SUCCESS)
        return 1;

    if (log) {
        size_t loglen = 0;

        clGetProgramBuildInfo(ocl->program, c->device, CL_PROGRAM_BUILD_LOG, 0,
                              NULL, &loglen);
        *log = G_malloc(loglen + 1);
        clGetProgramBuildInfo(ocl->program, c->device, CL_PROGRAM_BUILD_LOG,
                              loglen, *log, NULL);
        (*log)[loglen] = '\0';
    }
    clReleaseProgram(ocl->program);
    clReleaseCommandQueue(ocl->queue);
    clReleaseContext(ocl->context);
    ocl->program = NULL;
    ocl->queue = NULL;
    ocl->context = NULL;
    return 0;
}

void ocl_init(struct ocl_backend *ocl, const char *device_opt,
              const char *platform_opt, ocl_opts_fn opts, void *data)
{
    struct candidate cand[32];
    char *log = NULL;
    int n = 0, i;

    memset(ocl, 0, sizeof(*ocl));
    if (platform_opt && !*platform_opt)
        platform_opt = NULL;

    if (!device_opt || strcmp(device_opt, "auto") == 0) {
        n = list_devices(CL_DEVICE_TYPE_GPU, platform_opt, cand, n, 32);
        n = list_devices(CL_DEVICE_TYPE_CPU, platform_opt, cand, n, 32);
        n = list_devices(CL_DEVICE_TYPE_ALL, platform_opt, cand, n, 32);
    }
    else if (strcmp(device_opt, "gpu") == 0)
        n = list_devices(CL_DEVICE_TYPE_GPU, platform_opt, cand, n, 32);
    else if (strcmp(device_opt, "cpu") == 0)
        n = list_devices(CL_DEVICE_TYPE_CPU, platform_opt, cand, n, 32);
    else
        G_fatal_error(_("Invalid device option <%s>"), device_opt);

    if (n == 0)
        G_fatal_error(_("OpenCL: no usable %s device found%s%s"),
                      device_opt ? device_opt : "auto",
                      platform_opt ? _(" on platforms matching ") : "",
                      platform_opt ? platform_opt : "");

    /* The same GPU can be exposed by several platforms (e.g. Mesa Clover
     * and rusticl) of which only some compile the kernels: take the
     * first one that does. */
    for (i = 0; i < n; i++) {
        G_free(log);
        log = NULL;
        if (try_device(ocl, &cand[i], opts, data, &log))
            break;
        G_warning(_("OpenCL: kernels do not build on <%s> [%s], trying the "
                    "next device"),
                  ocl->device_name, ocl->platform_name);
        G_debug(1, "build log:\n%s", log);
    }
    if (i == n)
        G_fatal_error(_("OpenCL: kernel build failed on every candidate "
                        "device; last build log:\n%s"),
                      log ? log : "");
    G_free(log);

    G_message(_("OpenCL device: %s [%s], %lu MiB, max buffer %lu MiB"),
              ocl->device_name, ocl->platform_name,
              (unsigned long)(ocl->global_mem >> 20),
              (unsigned long)(ocl->max_alloc >> 20));
}

cl_command_queue ocl_queue(struct ocl_backend *ocl)
{
    cl_int err;
    cl_command_queue q =
        clCreateCommandQueue(ocl->context, ocl->device, 0, &err);

    ocl_check(err, "clCreateCommandQueue");
    return q;
}

void ocl_free(struct ocl_backend *ocl)
{
    if (ocl->program)
        clReleaseProgram(ocl->program);
    if (ocl->queue)
        clReleaseCommandQueue(ocl->queue);
    if (ocl->context)
        clReleaseContext(ocl->context);
    memset(ocl, 0, sizeof(*ocl));
}

cl_mem ocl_alloc(struct ocl_backend *ocl, size_t n, const char *what)
{
    cl_int err;
    cl_mem m;

    if (n == 0)
        n = 4;
    if (n > ocl->max_alloc)
        G_fatal_error(_("OpenCL: buffer <%s> needs %lu MiB, more than the "
                        "device maximum allocation of %lu MiB (reduce "
                        "batch=)"),
                      what, (unsigned long)(n >> 20),
                      (unsigned long)(ocl->max_alloc >> 20));
    m = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n, NULL, &err);
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: allocating %lu MiB for <%s> failed: %s"),
                      (unsigned long)(n >> 20), what, ocl_errstr(err));
    return m;
}

cl_mem ocl_upload(struct ocl_backend *ocl, const void *src, size_t n,
                  const char *what)
{
    cl_mem m = ocl_alloc(ocl, n, what);

    ocl_write(ocl, m, 0, src, n);
    return m;
}

void ocl_write(struct ocl_backend *ocl, cl_mem dst, size_t off, const void *src,
               size_t n)
{
    ocl_check(clEnqueueWriteBuffer(ocl->queue, dst, CL_TRUE, off, n, src, 0,
                                   NULL, NULL),
              "clEnqueueWriteBuffer");
}

void ocl_read(struct ocl_backend *ocl, cl_mem src, size_t off, void *dst,
              size_t n)
{
    ocl_check(clEnqueueReadBuffer(ocl->queue, src, CL_TRUE, off, n, dst, 0,
                                  NULL, NULL),
              "clEnqueueReadBuffer");
}

cl_kernel ocl_kernel(struct ocl_backend *ocl, const char *name)
{
    cl_int err;
    cl_kernel k = clCreateKernel(ocl->program, name, &err);

    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: kernel <%s> not found: %s"), name,
                      ocl_errstr(err));
    return k;
}

void ocl_release(cl_mem *m)
{
    if (*m) {
        clReleaseMemObject(*m);
        *m = NULL;
    }
}
