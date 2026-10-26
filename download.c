/****************************************************************************
 *
 * MODULE:       i.delineate.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Download of missing weights with libcurl: resumable,
 *               size-checked, atomic and safe against concurrent runs.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <curl/curl.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "download.h"

static long long file_size(const char *path)
{
    struct stat st;

    return stat(path, &st) == 0 && S_ISREG(st.st_mode) ? (long long)st.st_size
                                                       : -1;
}

/* mkdir -p of the folder holding path. */
static void make_parent(const char *path)
{
    char *dir = G_store(path), *p;

    if ((p = strrchr(dir, '/')))
        *p = '\0';
    else {
        G_free(dir);
        return;
    }
    for (p = dir + 1; *p; p++) {
        if (*p != '/')
            continue;
        *p = '\0';
        if (mkdir(dir, 0777) != 0 && errno != EEXIST)
            G_fatal_error(_("Unable to create folder <%s>: %s"), dir,
                          strerror(errno));
        *p = '/';
    }
    if (mkdir(dir, 0777) != 0 && errno != EEXIST)
        G_fatal_error(_("Unable to create folder <%s>: %s"), dir,
                      strerror(errno));
    G_free(dir);
}

struct progress {
    long long offset, total; /* Bytes already on disk, expected size. */
};

/* Report in MiB so that G_percent's long arithmetic never overflows. */
static int on_progress(void *data, curl_off_t dltotal, curl_off_t dlnow,
                       curl_off_t ultotal, curl_off_t ulnow)
{
    const struct progress *p = data;
    long long total = p->total ? p->total : p->offset + dltotal;

    (void)ultotal;
    (void)ulnow;
    if (total > 0)
        G_percent((long)((p->offset + dlnow) >> 20), (long)(total >> 20), 2);
    return 0;
}

void download_file(const char *url, const char *path, long long size)
{
    char *part, *lock, err[CURL_ERROR_SIZE];
    struct progress prog;
    long long have;
    CURL *curl;
    CURLcode rc;
    FILE *fp;
    int lfd;

    if (file_size(path) >= 0 && (size == 0 || file_size(path) == size))
        return;

    make_parent(path);
    G_asprintf(&part, "%s.part", path);
    G_asprintf(&lock, "%s.lock", path);

    lfd = open(lock, O_RDWR | O_CREAT, 0666);
    if (lfd < 0)
        G_fatal_error(_("Unable to write in the folder of <%s>: %s"), path,
                      strerror(errno));
    if (flock(lfd, LOCK_EX | LOCK_NB) != 0) {
        G_message(_("Waiting for another process downloading <%s>..."), path);
        if (flock(lfd, LOCK_EX) != 0)
            G_fatal_error(_("Unable to lock <%s>: %s"), lock, strerror(errno));
    }
    /* Another process may have completed it meanwhile. */
    if (file_size(path) >= 0 && (size == 0 || file_size(path) == size)) {
        close(lfd);
        G_free(part);
        G_free(lock);
        return;
    }

    have = file_size(part);
    if (have < 0 || (size && have >= size))
        have = 0;
    fp = fopen(part, have ? "ab" : "wb");
    if (!fp)
        G_fatal_error(_("Unable to write <%s>: %s"), part, strerror(errno));

    if (have)
        G_message(_("Resuming download of <%s> at %lld MiB..."), url,
                  have >> 20);
    else if (size)
        G_message(_("Downloading <%s> (%lld MiB) to <%s>..."), url, size >> 20,
                  path);
    else
        G_message(_("Downloading <%s> to <%s>..."), url, path);

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if (!curl)
        G_fatal_error(_("Unable to initialize libcurl"));
    prog.offset = have;
    prog.total = size;
    err[0] = '\0';
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, err);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "i.delineate.opencl");
    curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, (curl_off_t)have);
    /* Give up on a stalled transfer: below 1 KiB/s for a minute. */
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, on_progress);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &prog);
    rc = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    if (fclose(fp) != 0 && rc == CURLE_OK)
        G_fatal_error(_("Unable to write <%s>: %s"), part, strerror(errno));
    if (rc == CURLE_RANGE_ERROR)
        unlink(part); /* The server cannot resume: start over next time. */
    if (rc != CURLE_OK) {
        unlink(lock);
        G_fatal_error(_("Download of <%s> failed: %s (partial file kept in "
                        "<%s>, the next run resumes it)"),
                      url, err[0] ? err : curl_easy_strerror(rc), part);
    }
    G_percent(1, 1, 1);

    have = file_size(part);
    if (size && have != size) {
        unlink(part);
        unlink(lock);
        G_fatal_error(_("Download of <%s> has %lld bytes instead of %lld; "
                        "removed"),
                      url, have, size);
    }
    if (rename(part, path) != 0)
        G_fatal_error(_("Unable to rename <%s> to <%s>: %s"), part, path,
                      strerror(errno));
    unlink(lock);
    close(lfd);
    G_message(_("Weights saved as <%s>"), path);
    G_free(part);
    G_free(lock);
}
