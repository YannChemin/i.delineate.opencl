## DESCRIPTION

*i.delineate.opencl* delineates agricultural fields in a 3-band imagery
group with Delineate-Anything, a resolution-agnostic field boundary
model (Lavreniuk et al. 2025), and writes them as a vector area map. The
network, an Ultralytics YOLO11 instance segmentation model, runs on an
OpenCL device (GPU or CPU) with its weights read directly from the
published PyTorch file: neither Python nor PyTorch is needed at run
time. The detected field masks are then composed, merged across tiles
and filtered as the reference implementation's *main* method does, and
vectorized into a topologically clean map.

The **group** (or its **subgroup**) must contain exactly three raster
maps, used as the red, green and blue channels in the order they are
listed in the group (see *i.group*). Integer maps within 0-255 are used
as they are; other maps are stretched linearly between the lower and
upper **percentile** of their positive values to the 0-255 range the
model was trained on. Cells that are NULL in any band, or 0 in all three
(black, the reference's no-data value), hold no field. Processing is
done in the current computational region.

### Weights

The three models published by the authors are supported; the network
is built from the model description stored in the file, so the model is
recognised by its content:

| **model** | File | Network | Size |
| --------- | ---- | ------- | ---- |
| large_v2 (best quality) | [DelineateAnythingv2.pt](https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnythingv2.pt) | YOLO11x-seg | 125 MB |
| large | [DelineateAnything.pt](https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything.pt) | YOLO11x-seg | 125 MB |
| small (fastest) | [DelineateAnything-S.pt](https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything-S.pt) | YOLO11n-seg | 18 MB |

Without a **checkpoint** file, the module looks for the published file
names in the weights folder: **weights_dir**, by default
`$DELINEATE_MODELS_DIR`, or `~/models/delineate` when that environment
variable is not set. The best model present is used (large_v2, then
large, then small), or the one chosen with **model**. A missing model is
downloaded into that folder on first use (large_v2 when the folder holds
none); the download is checked against the published size, resumes
where an interrupted one stopped, and waits for another run downloading
the same file. The **-n** flag forbids downloading.

The pickled model is read without running any of its code: classes are
never imported, objects only become records of their class name and
attributes.

### Delineation

The model sees 512 x 512 pixel tiles. Imagery with cells of 4 m or more
is upsampled twice with OpenCV's bicubic kernel first (**scale**=*auto*;
1, 2 or 4 can be forced), so that a tile covers 256 x 256 region cells
of 10 m Sentinel-2 imagery; the fields are then delineated at half the
region resolution. Tiles overlap: they are placed every **tile_step**
times their size (half a tile by default) in the reference's order.

Each tile is run through the network. Detections above the
**confidence** threshold go through Ultralytics' non-maximum suppression
(IoU 0.7, at most 300) and their masks are built at full tile
resolution from the mask prototypes, then opened and closed with a
3 x 3 square. Of each mask, the largest outer contour is kept, filled,
and cut by the clip classes of the land cover **mask**. Detections
smaller than **min_pixels** tile cells, or lying for less than
**valid_fraction** of their area in valid data (at least 2 cells away
from no data) outside the filter classes, are dropped. Overlapping
detections of a tile are merged when their overlap (IoU) exceeds 0.8,
the larger ones placed first. Where tiles overlap, the smaller field
wins each cell, and fields of neighbouring tiles are merged into one
when they overlap enough (IoU above **merge_iou**, or within the 4-cell
edge band of a tile with an IoU above 0.6 or more than 192 cells in
common, and covering enough of each other). Finally the borders between
fields are cleared and the gaps filled from the neighbouring fields
(Delineate-Anything's identity opening), which leaves a thin line
between adjacent fields.

These steps and their defaults are those of the reference
implementation (`methods/main` with `conf_sample.yaml`).

### Land cover mask

An optional land cover **mask** (integer classes) refines the fields:
cells of the **clip_classes** (e.g. water, forest) are cut out of the
fields; a detection lying for more than 1 - **valid_fraction** of its
area in the **filter_classes** (e.g. built-up land) is dropped; and
cells of the **background_classes** where no field was found become
background areas, kept when larger than **background_min_area** with
their holes smaller than **background_min_hole** filled. NULL mask cells
belong to no class. The mask is read at the region resolution (nearest
neighbour).

### Filtering and vectorization

Holes in a field smaller than **min_hole** square meters are filled,
then fields smaller than **min_area** square meters are dropped. Both
thresholds are given for 10 m cells and scaled by the cell area divided
by 100 m² unless the **-u** flag is set (2500 m² for 10 m cells, 625 m²
for 5 m cells). Areas are computed from the cells, on the ellipsoid in
latitude-longitude projects. A small field lying in a filled hole is
absorbed; a field large enough to be kept stays an island.

Boundaries are traced along cell edges into shared topological
boundaries and simplified with the Douglas-Peucker algorithm with the
**simplify** tolerance, by default twice the cell diagonal of the region
(28 m for 10 m cells), the tolerance of the reference's simplification
step; 0 keeps the raster staircase. Junctions between fields are kept,
so neighbouring fields stay adjacent. The topology is then cleaned and
each field gets one centroid, placed at the cell farthest from its
border. The optional **raster** output holds the field categories before
simplification, at the delineation resolution.

### Attribute table

| Column     | Description                                              |
| ---------- | -------------------------------------------------------- |
| cat        | Field category                                           |
| background | 1 for a background area from the land cover mask, else 0 |
| class      | Land cover class of a background area, else 0            |
| area       | Area before simplification, in square meters             |

### OpenCL device

With **device**=*auto*, the first GPU is used, else the first CPU
device. When several OpenCL platforms expose the same device, the first
platform on which the kernels compile is used; **platform** restricts
the choice to platforms whose name contains the given text. Mesa's
rusticl only exposes a GPU when enabled in the environment, e.g.
`RUSTICL_ENABLE=radeonsi`.

### Device profile and precision

The **profile** adapts the run to the device:

- *cpu*: a CPU device such as PoCL. The whole network runs on the host
  over the mapped device buffers (no copy): convolutions as BLAS matrix
  products split into cache-sized blocks over the OpenMP threads, the
  other layers as OpenMP loops. PoCL runs the GPU-shaped kernels about
  ten times slower. Without a BLAS library at build time the OpenCL
  kernels are used.
- *igpu*: a GPU sharing host memory (integrated graphics, APU). The
  input and the network outputs live in host-visible memory and are
  written and read by mapping them (zero copy); the host prepares the
  next batch of tiles while the device runs the current one.
- *dgpu*: a GPU with its own memory on internal PCI Express. Tiles are
  uploaded from pinned host memory without blocking, and two batches
  are in flight.
- *egpu*: a GPU behind a slow link (Thunderbolt or USB4 enclosure,
  PoCL-Remote). As *dgpu*, with uploads and downloads on their own
  command queue so that they overlap the computation. The weights and
  activations stay on the device for the whole run.

With **profile**=*auto*, the profile is detected: *cpu* for a CPU
device, *igpu* for a GPU reporting unified memory, *egpu* for a remote
device or a GPU whose narrowest PCI Express link to the host carries
less than 6 GB/s (read from the driver's PCI address and Linux sysfs),
*dgpu* otherwise. The chosen profile and the reason are printed with
**--verbose**. A profile can be forced to compare them; *igpu* needs a
device sharing host memory.

**precision**=*fp16* stores the activations as IEEE half-precision
numbers, read and written with the core OpenCL `vload_half` and
`vstore_half` functions (no half-precision extension is needed); the
arithmetic stays in single precision. It halves the memory traffic of a
bandwidth-bound GPU, and is not available with *cpu*.
**precision**=*auto* uses fp32. fp16 moves boxes by less than 0.2 pixel
and scores by about 0.005, which is enough to tip detections lying at
the **confidence** threshold: on the Sentinel-2 sample, it gives 147
fields instead of 141, 130 of them matching the fp32 result with an
intersection over union above 0.95. Delineate-Anything itself runs in
half precision on GPUs, with the same effect.

### Memory

The fields are delineated over the whole region at once: the module
needs about 12 bytes per output cell (an identity and a weight per cell,
then the vectorization), plus 9 bytes per region cell for the imagery
and masks. A Sentinel-2 tile of 10980 x 10980 cells, delineated at 5 m,
needs about 6 GiB. Split larger areas with *g.region*. On the device,
the large models need about 1.2 GiB of activations for the default
**batch** of 4 tiles in fp32.

### Debugging aids

`I_DELINEATE_OPENCL_PROFILE=1` prints, per kernel (or host operation),
the calls, the time, and the achieved GFLOP/s and GB/s, which shows
whether it is limited by compute or by memory bandwidth; each kernel is
timed alone with the wall clock, which also works where driver event
times are wrong. When `I_DELINEATE_OPENCL_DUMP` names a directory, the
input and the output of every layer of the first tile, and every
detection, are written there for comparison with the PyTorch reference.
`I_DELINEATE_OPENCL_DOWNLOAD_URL` replaces the download location
(`https://huggingface.co/MykolaL/DelineateAnything/resolve/main`), for a
local mirror.

## NOTES

The port reproduces the reference results: on the Sentinel-2 sample
shipped with Delineate-Anything (1572 x 1540 cells at 10 m, small
model), it finds the same 141 fields, every one matching its reference
polygon with an intersection over union above 0.9 (median 0.9998)
before simplification. With the large_v2 model on a 768 x 768 cell
crop, all 111 fields match (intersection over union above 0.95, median
1.0). The network outputs match PyTorch to single precision (relative
error below 3e-4).

On a laptop Core i7-1165G7 (PoCL, *cpu* profile with OpenBLAS), the whole
sample takes 26 s with the small model, against about 4 minutes for the
reference implementation; the 768 x 768 crop takes 100 s with large_v2,
against 5 minutes 14 s. On the same laptop's Intel Iris Xe GPU
(*igpu* profile), the sample takes 46 s of delineation: the GEMM kernel
reaches only about 10 GFLOP/s there, so **device**=*cpu* is the faster
choice on such machines (`I_DELINEATE_OPENCL_PROFILE=1` shows the rate
reached on a given device).

Differences from the reference implementation:

- The region is processed as one block, so the reference's merging of
  polygons cut by its processing regions and worker blocks is not
  needed. Small holes are filled and fields filtered on the raster, in a
  topological map where polygons cannot overlap.
- The network runs in fp32 by default, on GPUs as well (the reference
  uses fp16 on GPUs).
- Merging tiles is done in the reference's tile order on one thread, so
  the result does not depend on the scheduling of parallel workers.
- The additional background vector source and multi-model passes of the
  reference configuration are not supported.

Delineate-Anything's code and weights are licensed under AGPL-3.0, and
so is this module.

## EXAMPLES

### Sentinel-2 true colour image

Delineate fields in a Sentinel-2 RGB composite at 10 m (the region is
upsampled twice, fields delineated at 5 m):

```sh
i.group group=s2_rgb input=B04,B03,B02
g.region raster=B04 -p
i.delineate.opencl group=s2_rgb output=fields
v.db.univar fields column=area
```

### Land cover mask

With a land cover map whose classes 1 and 2 are water and forest, 3 is
built-up land and 4 grassland, cut water and forest out of the fields,
drop fields in built-up areas and add grassland where no field is
found:

```sh
i.delineate.opencl group=s2_rgb output=fields mask=landcover \
    clip_classes=1,2 filter_classes=3 background_classes=4
```

### Fast preview on the CPU

```sh
i.delineate.opencl group=s2_rgb output=fields_quick model=small \
    device=cpu --verbose
```

## REFERENCES

Lavreniuk, M., Kussul, N., Shelestov, A., Yailymov, B., Salii, Y.,
Kuzin, V., Szantoi, Z., 2025. Delineate Anything: Resolution-Agnostic
Field Boundary Delineation on Satellite Imagery. ECAI 2025.
[arXiv:2504.02534](https://arxiv.org/abs/2504.02534)

Lavreniuk, M., Kussul, N., Shelestov, A., Salii, Y., Kuzin, V.,
Skakun, S., Szantoi, Z., 2025. Delineate Anything Flow: Fast,
Country-Level Field Boundary Detection from Any Source.
[arXiv:2511.13417](https://arxiv.org/abs/2511.13417)

Source code and models: <https://github.com/Lavreniuk/Delineate-Anything>

## SEE ALSO

*[i.group](i.group.md), [i.sam.opencl](i.sam.opencl.md),
[i.segment](i.segment.md), [r.to.vect](r.to.vect.md),
[v.clean](v.clean.md), [v.generalize](v.generalize.md)*

## AUTHORS

Yann Chemin, port of Delineate-Anything by Mykola Lavreniuk et al.
