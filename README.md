# i.delineate.opencl

GRASS GIS module that delineates agricultural fields in a 3-band imagery
group with [Delineate-Anything](https://github.com/Lavreniuk/Delineate-Anything)
(Lavreniuk et al. 2025) on an OpenCL device (GPU or CPU) and writes them
as a clean vector area map. The YOLO11 segmentation network runs in
OpenCL (or, on CPUs, as host BLAS products over the mapped device
buffers) and reads the original PyTorch weights directly: neither
Python nor PyTorch is needed at run time. The tiling, field composition,
merging across tiles and filtering of the reference implementation are
ported to C. See `i.delineate.opencl.md` for the full manual.

## Weights

The weights are not shipped with the module. The module downloads the
model it needs on first use into `$DELINEATE_MODELS_DIR` (by default
`~/models/delineate`), or into **weights_dir**; `-n` disables this. They
can also be fetched beforehand (AGPL-3.0):

| Model | File | Size |
| ----- | ---- | ---- |
| large_v2 (best quality) | `DelineateAnythingv2.pt` | 124 747 297 bytes |
| large | `DelineateAnything.pt` | 124 746 842 bytes |
| small (fastest) | `DelineateAnything-S.pt` | 17 635 629 bytes |

```sh
mkdir -p ~/models/delineate
cd ~/models/delineate
for f in DelineateAnythingv2.pt DelineateAnything.pt DelineateAnything-S.pt; do
    curl -L -O https://huggingface.co/MykolaL/DelineateAnything/resolve/main/$f
done
```

## Building

```sh
make MODULE_TOPDIR=$HOME/dev/grass
```

Requires an OpenCL 1.2 platform and its development headers (e.g.
`ocl-icd-opencl-dev`), libcurl (`libcurl4-openssl-dev`) and a GRASS GIS
build with OpenCL support. A GRASS build with BLAS (OpenBLAS) gives the
fast CPU path. With Mesa on AMD GPUs, export `RUSTICL_ENABLE=radeonsi`
(or the matching driver) to expose the GPU through rusticl.

## Tests

```sh
I_DELINEATE_OPENCL_CHECKPOINT=~/models/delineate/DelineateAnything-S.pt pytest tests
```

Without `I_DELINEATE_OPENCL_CHECKPOINT`, only the input validation tests
run.

## Licence

Delineate-Anything's code and weights are licensed under the GNU Affero
General Public License v3.0 (AGPL-3.0), and so is this port (`LICENSE`).
The generic helpers shared with the other `*.opencl` modules (OpenCL
setup, checkpoint reader, download, progress) are also released under
the Unlicense, as stated in their headers.
