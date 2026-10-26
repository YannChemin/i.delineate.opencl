/****************************************************************************
 *
 * MODULE:       i.delineate.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Whole-run progress reporting. Every stage plans its work
 *               up front, and G_percent() sees a single rising count, so
 *               the GRASS bar and MetaMonitor's GRASS hook
 *               (https://github.com/YannChemin/MetaMonitor) show one bar
 *               for the run instead of one 0-100 % bar per stage.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <time.h>

#include <grass/gis.h>

#include "progress.h"

/* Seconds per unit used until a kind has been timed. The tile figure is
 * a slow CPU one on purpose: on a faster device the first finished tiles
 * correct it and the bar jumps forward, while an optimistic guess would
 * let the quick early stages fill the bar and then stall it through the
 * inference. */
static const double prior[PG_NKINDS] = {
    0.5,  /* PG_READ, per million cells. */
    6.0,  /* PG_LOAD, per GiB. */
    3.0,  /* PG_BUILD. */
    20.0, /* PG_INFER, per tile. */
    2.0,  /* PG_MERGE. */
    5.0,  /* PG_VECTOR. */
};

static struct {
    double planned[PG_NKINDS], done[PG_NKINDS], seconds[PG_NKINDS];
    int cur;          /* Open kind, or -1. */
    double cur_units; /* Units of the open stage. */
    double frac;      /* Fraction of the open stage done. */
    double t0;        /* Start time of the open stage. */
    double load_total, load_read;
    int last; /* Last reported per mille. */
} pg = {{0}, {0}, {0}, -1, 0, 0, 0, 0, 0, -1};

static double now(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int is_device(int k) { return k == PG_INFER; }

/* Seconds per unit of kind k measured so far, counting the open stage
 * once part of it is done; 0 when nothing has been timed. */
static double measured(int k)
{
    double units = pg.done[k], secs = pg.seconds[k];

    if (k == pg.cur && pg.frac > 0) {
        units += pg.cur_units * pg.frac;
        secs += now() - pg.t0;
    }
    return units > 0 && secs > 0 ? secs / units : 0;
}

/* Seconds per unit of kind k, measured or estimated. */
static double cost(int k)
{
    double speed = 0, n = 0, c = measured(k);
    int j;

    if (c > 0 || !is_device(k))
        return c > 0 ? c : prior[k];
    for (j = 0; j < PG_NKINDS; j++)
        if (is_device(j) && (c = measured(j)) > 0) {
            speed += c / prior[j];
            n++;
        }
    return n > 0 ? prior[k] * speed / n : prior[k];
}

static void report(void)
{
    double done = 0, total = 0;
    int k, pm;

    for (k = 0; k < PG_NKINDS; k++) {
        double c = cost(k);

        total += pg.planned[k] * c;
        done += pg.done[k] * c;
    }
    if (pg.cur >= 0)
        done += pg.cur_units * pg.frac * cost(pg.cur);
    if (total <= 0)
        return;
    pm = (int)(1000 * done / total);
    /* The bar never goes back when estimates change, and only
     * progress_finish() reaches 100 %, which resets G_percent(). */
    if (pm > 999)
        pm = 999;
    if (pm <= pg.last)
        return;
    pg.last = pm;
    G_percent(pm, 1000, 1);
}

void progress_plan(enum progress_kind k, double units)
{
    pg.planned[k] += units;
    report();
}

void progress_begin(enum progress_kind k, double units)
{
    if (pg.cur >= 0)
        progress_end();
    /* Unplanned work extends the plan. */
    if (pg.done[k] + units > pg.planned[k])
        pg.planned[k] = pg.done[k] + units;
    pg.cur = k;
    pg.cur_units = units;
    pg.frac = 0;
    pg.t0 = now();
    if (k == PG_LOAD) {
        pg.load_total = units * 1073741824.0;
        pg.load_read = 0;
    }
    report();
}

void progress_part(double done, double total)
{
    if (pg.cur < 0 || total <= 0)
        return;
    pg.frac = done < total ? done / total : 1;
    report();
}

void progress_bytes(size_t n)
{
    if (pg.cur != PG_LOAD)
        return;
    pg.load_read += n;
    progress_part(pg.load_read, pg.load_total);
}

void progress_end(void)
{
    if (pg.cur < 0)
        return;
    pg.done[pg.cur] += pg.cur_units;
    pg.seconds[pg.cur] += now() - pg.t0;
    pg.cur = -1;
    report();
}

void progress_finish(void)
{
    if (pg.cur >= 0)
        progress_end();
    G_percent(1, 1, 1);
}
