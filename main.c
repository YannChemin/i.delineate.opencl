/****************************************************************************
 *
 * MODULE:       i.delineate.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Delineate agricultural field boundaries in a 3-band
 *               imagery group with Delineate-Anything (Lavreniuk et al.
 *               2025), a YOLO11 instance segmentation network run on an
 *               OpenCL device from the original PyTorch weights, and
 *               write the fields as a vector area map.
 *
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 *****************************************************************************/

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "delineate.h"
#include "download.h"
#include "image.h"
#include "ocl_backend.h"
#include "progress.h"
#include "pth_loader.h"
#include "vectorize.h"
#include "yolo_model.h"

static double now(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* Delineate-Anything models in order of preference, as published by the
 * authors (AGPL-3.0), with their sizes in bytes. */
#define DOWNLOAD_URL                                                           \
    "https://huggingface.co/MykolaL/DelineateAnything/resolve/main"

static const struct {
    const char *name, *file;
    long long size;
} models[] = {
    {"large_v2", "DelineateAnythingv2.pt", 124747297LL},
    {"large", "DelineateAnything.pt", 124746842LL},
    {"small", "DelineateAnything-S.pt", 17635629LL},
};

#define N_MODELS ((int)(sizeof(models) / sizeof(models[0])))

/* The central folder of the weights: $DELINEATE_MODELS_DIR, by default
 * ~/models/delineate. */
static char *default_weights_dir(void)
{
    const char *base = getenv("DELINEATE_MODELS_DIR");
    char *dir;

    if (base && *base)
        dir = G_store(base);
    else
        G_asprintf(&dir, "%s/models/delineate", G_home());
    return dir;
}

static int model_index(const char *name)
{
    int v;

    for (v = 0; v < N_MODELS; v++)
        if (strcmp(models[v].name, name) == 0)
            return v;
    G_fatal_error(_("Unknown model <%s>"), name);
    return -1;
}

/* Path of the weights of model v in dir if present, else NULL. */
static char *weights_in(const char *dir, int v)
{
    struct stat st;
    char *path;

    G_asprintf(&path, "%s/%s", dir, models[v].file);
    if (stat(path, &st) == 0 && S_ISREG(st.st_mode))
        return path;
    G_free(path);
    return NULL;
}

/* Download model v into dir; returns its path. I_DELINEATE_OPENCL_DOWNLOAD_URL
 * replaces the download location (a mirror). */
static char *fetch_weights(const char *dir, int v)
{
    const char *base = getenv("I_DELINEATE_OPENCL_DOWNLOAD_URL");
    char *url, *path;

    if (!base || !*base)
        base = DOWNLOAD_URL;
    G_asprintf(&url, "%s/%s", base, models[v].file);
    G_asprintf(&path, "%s/%s", dir, models[v].file);
    download_file(url, path, models[v].size);
    G_free(url);
    return path;
}

static int *parse_ints(const struct Option *opt, int *n)
{
    int *v, i;

    *n = 0;
    if (!opt->answers)
        return NULL;
    while (opt->answers[*n])
        (*n)++;
    v = G_malloc(sizeof(int) * *n);
    for (i = 0; i < *n; i++)
        v[i] = atoi(opt->answers[i]);
    return v;
}

/* Cell size of the current region in meters (mean of both axes) and
 * cell area in square meters, as Delineate-Anything estimates them:
 * east-west cells of a latitude-longitude region are measured at the
 * region's central latitude on a sphere. */
static void cell_size_m(const struct Cell_head *win, double *size, double *area)
{
    double w, h;

    if (win->proj == PROJECTION_LL) {
        double lat_m = 6371000.0 * M_PI / 180.0;
        double lat = (win->north + win->south) / 2 * M_PI / 180.0;

        w = win->ew_res * lat_m * cos(lat);
        h = win->ns_res * lat_m;
    }
    else {
        double f = win->proj == PROJECTION_XY
                       ? 1.0
                       : G_database_units_to_meters_factor();

        if (win->proj == PROJECTION_XY)
            G_warning(_("Unprojected (XY) project: map units are taken as "
                        "meters"));
        w = win->ew_res * f;
        h = win->ns_res * f;
    }
    *size = (w + h) / 2;
    *area = w * h;
}

int main(int argc, char *argv[])
{
    struct GModule *module;
    struct {
        struct Option *group, *subgroup, *checkpoint, *weights_dir, *model,
            *output, *raster, *mask, *clip, *filter, *bg, *percentile, *scale,
            *conf, *step, *min_area, *min_hole, *bg_min_area, *bg_min_hole,
            *simplify, *min_pixels, *valid_fraction, *merge_iou, *batch,
            *device, *platform, *profile, *precision;
    } opt;
    struct {
        struct Flag *offline, *fixed;
    } flag;
    struct ocl_backend ocl;
    struct pth_file pf;
    struct yolo_model model;
    struct del_image img;
    struct del_mask mask;
    struct del_params dp;
    struct vect_params vp;
    struct yolo_setup setup;
    struct Cell_head win;
    double lo, hi, t0, t1, load_gib, px_m, px_area, area_coeff = 1.0;
    char source[GNAME_MAX * 2 + 32], *desc, *checkpoint, *dump;
    int *clip_cls, *filter_cls, *bg_cls, nclip, nfilter, nbg, *ids, v;
    long ntiles;

    G_gisinit(argv[0]);

    module = G_define_module();
    G_add_keyword(_("imagery"));
    G_add_keyword(_("agriculture"));
    G_add_keyword(_("field boundaries"));
    G_add_keyword(_("segmentation"));
    G_add_keyword(_("deep learning"));
    G_add_keyword(_("GPU"));
    G_add_keyword(_("OpenCL"));
    G_add_keyword(_("vectorization"));
    module->description =
        _("Delineates agricultural fields in a 3-band imagery group with "
          "Delineate-Anything on an OpenCL device and writes them as vector "
          "areas.");

    opt.group = G_define_standard_option(G_OPT_I_GROUP);
    opt.group->description =
        _("Name of input imagery group with exactly 3 raster maps (R, G, B)");

    opt.subgroup = G_define_standard_option(G_OPT_I_SUBGROUP);
    opt.subgroup->required = NO;
    opt.subgroup->description =
        _("Name of subgroup with exactly 3 raster maps to use instead of "
          "the whole group");

    opt.output = G_define_standard_option(G_OPT_V_OUTPUT);
    opt.output->description = _("Name for output vector map of fields");

    opt.raster = G_define_standard_option(G_OPT_R_OUTPUT);
    opt.raster->key = "raster";
    opt.raster->required = NO;
    opt.raster->description =
        _("Name for optional output raster map of field categories, at the "
          "processing resolution");

    opt.checkpoint = G_define_standard_option(G_OPT_F_INPUT);
    opt.checkpoint->key = "checkpoint";
    opt.checkpoint->required = NO;
    opt.checkpoint->label = _("Delineate-Anything weights file");
    opt.checkpoint->description =
        _("Original PyTorch .pt file, e.g. DelineateAnythingv2.pt");
    opt.checkpoint->guisection = _("Model");

    opt.weights_dir = G_define_standard_option(G_OPT_M_DIR);
    opt.weights_dir->key = "weights_dir";
    opt.weights_dir->required = NO;
    opt.weights_dir->label = _("Folder of Delineate-Anything weights");
    opt.weights_dir->description =
        _("Folder of the published .pt files, where missing ones are "
          "downloaded; default: $DELINEATE_MODELS_DIR, else "
          "~/models/delineate");
    opt.weights_dir->guisection = _("Model");

    opt.model = G_define_option();
    opt.model->key = "model";
    opt.model->type = TYPE_STRING;
    opt.model->required = NO;
    opt.model->options = "large_v2,large,small";
    opt.model->label = _("Model to use from the weights folder");
    opt.model->description =
        _("Default: the best one present (large_v2, then large, then "
          "small), or large_v2 if there is none");
    G_asprintf(&desc, "large_v2;%s;large;%s;small;%s",
               _("Delineate Anything v2 (YOLO11x), 125 MB, best quality"),
               _("Delineate Anything (YOLO11x), 125 MB"),
               _("Delineate Anything S (YOLO11n), 18 MB, fastest"));
    opt.model->descriptions = desc;
    opt.model->guisection = _("Model");

    opt.mask = G_define_standard_option(G_OPT_R_INPUT);
    opt.mask->key = "mask";
    opt.mask->required = NO;
    opt.mask->label = _("Land cover raster map");
    opt.mask->description =
        _("Integer classes used with clip_classes, filter_classes and "
          "background_classes");
    opt.mask->guisection = _("Land cover");

    opt.clip = G_define_option();
    opt.clip->key = "clip_classes";
    opt.clip->type = TYPE_INTEGER;
    opt.clip->multiple = YES;
    opt.clip->required = NO;
    opt.clip->description =
        _("Land cover classes cut out of the fields (e.g. water, forest)");
    opt.clip->guisection = _("Land cover");

    opt.filter = G_define_option();
    opt.filter->key = "filter_classes";
    opt.filter->type = TYPE_INTEGER;
    opt.filter->multiple = YES;
    opt.filter->required = NO;
    opt.filter->description =
        _("Land cover classes in which a field may lie for less than "
          "1 - valid_fraction of its area (e.g. built-up)");
    opt.filter->guisection = _("Land cover");

    opt.bg = G_define_option();
    opt.bg->key = "background_classes";
    opt.bg->type = TYPE_INTEGER;
    opt.bg->multiple = YES;
    opt.bg->required = NO;
    opt.bg->description =
        _("Land cover classes added as background areas where no field is "
          "found");
    opt.bg->guisection = _("Land cover");

    opt.percentile = G_define_option();
    opt.percentile->key = "percentile";
    opt.percentile->type = TYPE_DOUBLE;
    opt.percentile->key_desc = "lower,upper";
    opt.percentile->answer = "1,99";
    opt.percentile->options = "0-100";
    opt.percentile->description =
        _("Percentiles of the positive values of each band stretched to "
          "0-255 (not for 8-bit bands)");
    opt.percentile->guisection = _("Input");

    opt.scale = G_define_option();
    opt.scale->key = "scale";
    opt.scale->type = TYPE_STRING;
    opt.scale->options = "auto,1,2,4";
    opt.scale->answer = "auto";
    opt.scale->label = _("Upsampling factor of the imagery before "
                         "delineation");
    opt.scale->description =
        _("auto: 1 for cells smaller than 4 m, else 2; the fields are "
          "delineated at the region resolution divided by this factor");
    opt.scale->guisection = _("Input");

    opt.conf = G_define_option();
    opt.conf->key = "confidence";
    opt.conf->type = TYPE_DOUBLE;
    opt.conf->answer = "0.15";
    opt.conf->options = "0-1";
    opt.conf->description = _("Minimum detection confidence of a field");
    opt.conf->guisection = _("Delineation");

    opt.step = G_define_option();
    opt.step->key = "tile_step";
    opt.step->type = TYPE_DOUBLE;
    opt.step->answer = "0.5";
    opt.step->options = "0.05-1";
    opt.step->description =
        _("Step between tiles as a fraction of the tile size");
    opt.step->guisection = _("Delineation");

    opt.min_pixels = G_define_option();
    opt.min_pixels->key = "min_pixels";
    opt.min_pixels->type = TYPE_INTEGER;
    opt.min_pixels->answer = "512";
    opt.min_pixels->options = "0-";
    opt.min_pixels->description =
        _("Minimum size of a detected field in cells of the 512 x 512 "
          "network tile");
    opt.min_pixels->guisection = _("Delineation");

    opt.valid_fraction = G_define_option();
    opt.valid_fraction->key = "valid_fraction";
    opt.valid_fraction->type = TYPE_DOUBLE;
    opt.valid_fraction->answer = "0.8";
    opt.valid_fraction->options = "0-1";
    opt.valid_fraction->description =
        _("Minimum fraction of a detected field inside valid data and "
          "outside the filter classes");
    opt.valid_fraction->guisection = _("Delineation");

    opt.merge_iou = G_define_option();
    opt.merge_iou->key = "merge_iou";
    opt.merge_iou->type = TYPE_DOUBLE;
    opt.merge_iou->answer = "0.8";
    opt.merge_iou->options = "0-1";
    opt.merge_iou->description =
        _("Overlap (IoU) above which detections are merged into one field");
    opt.merge_iou->guisection = _("Delineation");

    opt.min_area = G_define_option();
    opt.min_area->key = "min_area";
    opt.min_area->type = TYPE_DOUBLE;
    opt.min_area->answer = "2500";
    opt.min_area->options = "0-";
    opt.min_area->description =
        _("Minimum field area in square meters (for 10 m cells, see -u)");
    opt.min_area->guisection = _("Filtering");

    opt.min_hole = G_define_option();
    opt.min_hole->key = "min_hole";
    opt.min_hole->type = TYPE_DOUBLE;
    opt.min_hole->answer = "2500";
    opt.min_hole->options = "0-";
    opt.min_hole->description =
        _("Holes in a field smaller than this, in square meters (for 10 m "
          "cells, see -u), are filled");
    opt.min_hole->guisection = _("Filtering");

    opt.bg_min_area = G_define_option();
    opt.bg_min_area->key = "background_min_area";
    opt.bg_min_area->type = TYPE_DOUBLE;
    opt.bg_min_area->answer = "50000";
    opt.bg_min_area->options = "0-";
    opt.bg_min_area->description =
        _("Minimum background area in square meters");
    opt.bg_min_area->guisection = _("Filtering");

    opt.bg_min_hole = G_define_option();
    opt.bg_min_hole->key = "background_min_hole";
    opt.bg_min_hole->type = TYPE_DOUBLE;
    opt.bg_min_hole->answer = "25000";
    opt.bg_min_hole->options = "0-";
    opt.bg_min_hole->description =
        _("Holes in a background area smaller than this, in square meters, "
          "are filled");
    opt.bg_min_hole->guisection = _("Filtering");

    opt.simplify = G_define_option();
    opt.simplify->key = "simplify";
    opt.simplify->type = TYPE_DOUBLE;
    opt.simplify->options = "0-";
    opt.simplify->label = _("Boundary simplification tolerance in map units");
    opt.simplify->description =
        _("Douglas-Peucker tolerance; default is twice the cell diagonal of "
          "the region, 0 keeps the raster staircase");
    opt.simplify->guisection = _("Filtering");

    opt.batch = G_define_option();
    opt.batch->key = "batch";
    opt.batch->type = TYPE_INTEGER;
    opt.batch->answer = "4";
    opt.batch->options = "1-64";
    opt.batch->description =
        _("Number of tiles processed at once on the device");
    opt.batch->guisection = _("Processing");

    opt.device = G_define_option();
    opt.device->key = "device";
    opt.device->type = TYPE_STRING;
    opt.device->options = "auto,gpu,cpu";
    opt.device->answer = "auto";
    opt.device->description = _("OpenCL device type");
    opt.device->guisection = _("Processing");

    opt.platform = G_define_option();
    opt.platform->key = "platform";
    opt.platform->type = TYPE_STRING;
    opt.platform->required = NO;
    opt.platform->description =
        _("Use only OpenCL platforms whose name contains this text");
    opt.platform->guisection = _("Processing");

    opt.profile = G_define_option();
    opt.profile->key = "profile";
    opt.profile->type = TYPE_STRING;
    opt.profile->options = "auto,cpu,igpu,dgpu,egpu";
    opt.profile->answer = "auto";
    opt.profile->label = _("Device profile");
    opt.profile->description =
        _("How the network uses the device; auto detects it");
    G_asprintf(&desc, "auto;%s;cpu;%s;igpu;%s;dgpu;%s;egpu;%s",
               _("from the device type, memory and host link"),
               _("CPU device: convolutions as host BLAS products"),
               _("GPU sharing memory with the host: zero-copy input and "
                 "outputs"),
               _("discrete GPU: pinned transfers, two batches in flight"),
               _("GPU behind a slow link (Thunderbolt, remote): transfers "
                 "overlapped with the compute"));
    opt.profile->descriptions = desc;
    opt.profile->guisection = _("Processing");

    opt.precision = G_define_option();
    opt.precision->key = "precision";
    opt.precision->type = TYPE_STRING;
    opt.precision->options = "auto,fp32,fp16";
    opt.precision->answer = "auto";
    opt.precision->label = _("Storage precision of the activations");
    opt.precision->description =
        _("auto: fp32; fp16 halves the activation memory and traffic on "
          "GPU profiles but can change detections near the confidence "
          "threshold; arithmetic is always fp32");
    opt.precision->guisection = _("Processing");

    flag.offline = G_define_flag();
    flag.offline->key = 'n';
    flag.offline->description =
        _("Do not download missing weights into the weights folder");
    flag.offline->guisection = _("Model");

    flag.fixed = G_define_flag();
    flag.fixed->key = 'u';
    flag.fixed->label = _("Use min_area and min_hole as given");
    flag.fixed->description =
        _("By default they are scaled by the cell area relative to 10 m "
          "cells");
    flag.fixed->guisection = _("Filtering");

    G_option_exclusive(opt.checkpoint, opt.weights_dir, NULL);
    G_option_exclusive(opt.checkpoint, opt.model, NULL);
    G_option_requires(opt.clip, opt.mask, NULL);
    G_option_requires(opt.filter, opt.mask, NULL);
    G_option_requires(opt.bg, opt.mask, NULL);
    G_option_requires(opt.mask, opt.clip, opt.filter, opt.bg, NULL);

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    lo = atof(opt.percentile->answers[0]);
    hi = opt.percentile->answers[1] ? atof(opt.percentile->answers[1]) : -1;
    if (!(hi > lo))
        G_fatal_error(_("<%s> needs two increasing values"),
                      opt.percentile->key);

    G_get_window(&win);
    cell_size_m(&win, &px_m, &px_area);
    memset(&dp, 0, sizeof(dp));
    if (strcmp(opt.scale->answer, "auto") == 0)
        dp.scale = px_m < 4 ? 1 : 2;
    else
        dp.scale = atoi(opt.scale->answer);
    G_message(_("Cell size %.2f m: delineating at %d x the region "
                "resolution"),
              px_m, dp.scale);
    dp.tile_step = atof(opt.step->answer);
    dp.conf = (float)atof(opt.conf->answer);
    dp.nms_iou = 0.7f;
    dp.max_det = 300;
    dp.batch = atoi(opt.batch->answer);
    dp.pixel_area_threshold = atoi(opt.min_pixels->answer);
    dp.remaining_area_threshold = atof(opt.valid_fraction->answer);
    dp.compose_merge_iou = 0.8;
    dp.merge_iou = atof(opt.merge_iou->answer);
    dp.merge_relative_area_threshold = 0.5;
    dp.merge_asymetric_pixel_area_threshold = 32;
    dp.merge_asymetric_relative_area_threshold = 0.7;
    dp.merging_edge_width = 4;
    dp.merge_edge_iou = 0.6;
    dp.merge_edge_pixels = 192;
    /* Developer aid: I_DELINEATE_OPENCL_DUMP=dir writes the network
     * outputs of the first batch and every detection. */
    dump = getenv("I_DELINEATE_OPENCL_DUMP");
    dp.dump_dir = dump && *dump ? dump : NULL;

    if (!flag.fixed->answer)
        area_coeff = px_area / 100.0;
    vp.scale = dp.scale;
    vp.min_area = atof(opt.min_area->answer) * area_coeff;
    vp.min_hole = atof(opt.min_hole->answer) * area_coeff;
    vp.bg_min_area = atof(opt.bg_min_area->answer);
    vp.bg_min_hole = atof(opt.bg_min_hole->answer);
    vp.simplify = opt.simplify->answer ? atof(opt.simplify->answer)
                                       : 2.0 * hypot(win.ew_res, win.ns_res);
    vp.vector_out = opt.output->answer;
    vp.raster_out = opt.raster->answer;
    snprintf(source, sizeof(source), "i.delineate.opencl group=%s%s%s",
             opt.group->answer, opt.subgroup->answer ? " subgroup=" : "",
             opt.subgroup->answer ? opt.subgroup->answer : "");
    vp.source = source;
    G_verbose_message(_("Minimum field area %g m2, minimum hole %g m2"),
                      vp.min_area, vp.min_hole);

    clip_cls = parse_ints(opt.clip, &nclip);
    filter_cls = parse_ints(opt.filter, &nfilter);
    bg_cls = parse_ints(opt.bg, &nbg);
    mask.name = opt.mask->answer;
    mask.clip = clip_cls;
    mask.nclip = nclip;
    mask.filter = filter_cls;
    mask.nfilter = nfilter;
    mask.bg = bg_cls;
    mask.nbg = nbg;
    for (v = 0; v < nbg; v++)
        if (bg_cls[v] <= 0)
            G_fatal_error(_("Background classes must be positive"));

    /* One progress bar for the whole run: plan the work that the region
     * and the options already determine; the weights are added once they
     * are open. */
    img.rows = Rast_window_rows();
    img.cols = Rast_window_cols();
    ntiles = delineate_count_tiles(&img, &dp);
    progress_plan(PG_READ, (double)img.rows * img.cols / 1e6);
    progress_plan(PG_BUILD, 1);
    progress_plan(PG_INFER, ntiles);
    progress_plan(PG_MERGE, 1);
    progress_plan(PG_VECTOR, 1);

    /* Validate the cheap inputs (group, weights index) before the OpenCL
     * device is initialized and the weights are uploaded. */
    image_read_group(opt.group->answer, opt.subgroup->answer, lo, hi, &img);
    image_read_mask(&mask, &img);
    if (opt.checkpoint->answer)
        checkpoint = G_store(opt.checkpoint->answer);
    else {
        char *dir = opt.weights_dir->answer ? G_store(opt.weights_dir->answer)
                                            : default_weights_dir();

        checkpoint = NULL;
        if (opt.model->answer) {
            v = model_index(opt.model->answer);
            checkpoint = weights_in(dir, v);
        }
        else
            for (v = 0; v < N_MODELS && !checkpoint; v++)
                checkpoint = weights_in(dir, v);
        if (!checkpoint) {
            v = opt.model->answer ? model_index(opt.model->answer) : 0;
            if (flag.offline->answer)
                G_fatal_error(_("No %s weights (%s) in weights folder <%s>"),
                              models[v].name, models[v].file, dir);
            checkpoint = fetch_weights(dir, v);
        }
        G_message(_("Using weights <%s>"), checkpoint);
        G_free(dir);
    }
    pth_open(&pf, checkpoint);
    G_free(checkpoint);
    load_gib = pth_bytes(&pf, "") / 1073741824.0;
    progress_plan(PG_LOAD, load_gib);

    t0 = now();
    progress_begin(PG_BUILD, 1);
    setup.profile_opt = opt.profile->answer;
    setup.precision_opt = opt.precision->answer;
    ocl_init(&ocl, opt.device->answer, opt.platform->answer, yolo_device_opts,
             &setup);
    G_message(_("Device profile: %s, %s activations"),
              yolo_profile_names[setup.profile], setup.half ? "fp16" : "fp32");
    progress_begin(PG_LOAD, load_gib);
    yolo_load(&model, &ocl, &setup, &pf, dp.batch);
    progress_end();
    pth_close(&pf);
    t1 = now();
    G_message(_("Model ready in %.1f s"), t1 - t0);

    ids = delineate_run(&model, &img, &dp);
    t0 = now();
    G_message(_("Delineation took %.1f s"), t0 - t1);

    if (ocl.profile)
        yolo_report(&model);
    yolo_free(&model);
    ocl_free(&ocl);

    vectorize_fields(ids, img.rows * dp.scale, img.cols * dp.scale, &vp);
    progress_finish();
    G_message(_("Vectorization took %.1f s"), now() - t0);

    image_free(&img);
    G_free(clip_cls);
    G_free(filter_cls);
    G_free(bg_cls);

    exit(EXIT_SUCCESS);
}
