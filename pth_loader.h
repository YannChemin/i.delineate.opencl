/* SPDX-License-Identifier: Unlicense */

#ifndef I_DELINEATE_OPENCL_PTH_LOADER_H
#define I_DELINEATE_OPENCL_PTH_LOADER_H

#include <stdio.h>
#include <sys/types.h>

#define PTH_MAX_DIMS 8

enum pth_dtype { PTH_F32, PTH_F16, PTH_BF16, PTH_OTHER };

/* One tensor of a PyTorch state_dict, described by where its storage
 * lives inside the (uncompressed) zip archive and how it is strided
 * over that storage. */
struct pth_tensor {
    char *name;
    int ndim;
    long shape[PTH_MAX_DIMS];
    long stride[PTH_MAX_DIMS];
    long storage_offset; /* In elements. */
    enum pth_dtype dtype;
    off_t data_offset;    /* Byte offset of the storage in the file. */
    size_t storage_bytes; /* Byte size of the storage entry. */
};

/* A scalar or small integer sequence attribute of a pickled module,
 * e.g. Conv2d.stride or Bottleneck.add. */
enum pth_attr_type { PA_INT, PA_FLOAT, PA_STR, PA_INTS };

struct pth_attr {
    char *name;
    enum pth_attr_type type;
    long v[PTH_MAX_DIMS]; /* PA_INT (v[0], booleans as 0/1) and PA_INTS. */
    int n;                /* Number of PA_INTS values. */
    double f;
    char *s;
};

/* One torch.nn.Module of a checkpoint that pickles a whole model, named
 * by its path in the state_dict ("" for the root, then e.g. "model.0"). */
struct pth_module {
    char *path;
    char *cls; /* Class name without its Python module, e.g. "Conv2d". */
    int nattr;
    struct pth_attr *attr;
};

struct pth_file {
    FILE *fp;
    char *path;
    int ntensors;
    struct pth_tensor *tensors;
    int nmodules; /* 0 for a checkpoint holding a bare state_dict. */
    struct pth_module *modules;
};

/* Open a PyTorch checkpoint (torch.save zip format, stored entries) and
 * index its floating-point tensors: a state_dict, possibly under a
 * "model" or "state_dict" key, or a whole pickled model under "model" or
 * (when "model" is None, as in training checkpoints) "ema", whose module
 * tree is then indexed too. Classes are never imported or run: pickled
 * objects only become records of their class name and attributes. Calls
 * G_fatal_error() on any format problem. */
void pth_open(struct pth_file *pf, const char *path);

/* Return the module at path, or NULL. */
const struct pth_module *pth_module(const struct pth_file *pf,
                                    const char *path);

/* Return the attribute called name of module m, or NULL. */
const struct pth_attr *pth_attr(const struct pth_module *m, const char *name);

/* Return the tensor called name, or NULL if absent. */
const struct pth_tensor *pth_find(const struct pth_file *pf, const char *name);

/* Like pth_find(), but fatal if the tensor is missing or its shape does
 * not match the ndim leading entries of shape[] (-1 matches anything). */
const struct pth_tensor *pth_require(const struct pth_file *pf,
                                     const char *name, int ndim,
                                     const long *shape);

long pth_numel(const struct pth_tensor *t);

/* Total storage bytes of the tensors whose names start with prefix. */
double pth_bytes(const struct pth_file *pf, const char *prefix);

/* Read a tensor as a newly allocated, C-contiguous float32 array. */
float *pth_read_f32(const struct pth_file *pf, const struct pth_tensor *t);

void pth_close(struct pth_file *pf);

#endif /* I_DELINEATE_OPENCL_PTH_LOADER_H */
