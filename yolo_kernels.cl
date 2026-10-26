/* OpenCL C 1.1 kernels for i.delineate.opencl: the layers of a YOLO11
 * segmentation network.
 *
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 * Activations are channel-major (C, H, W) planes, one image after the
 * other, stored as float32, or as float16 when ACT_HALF is defined (the
 * fp16 precision: storage only, arithmetic stays float32, so cl_khr_fp16
 * is not needed). A tensor argument is a buffer, an element offset and a batch
 * stride (elements from one image to the next), so that a kernel can read
 * or write a channel slice of a larger buffer, as the concatenations of
 * the network need. The third global dimension is the image of the batch,
 * unless stated otherwise. */

/* Work-group of WG x WG work-items, each computing RB x RB outputs: a
 * work-group computes a TILE_M (output channels) x TILE_N (pixels) block,
 * stepping through the reduction by TILE_K. The host defines WG, RB and
 * TILE_K at build time (YOLO_BUILD_OPTS in yolo_model.h). WG * WG must be
 * a multiple of TILE_N and TILE_K a multiple of WG * WG / TILE_N. */
#if !defined(WG) || !defined(RB) || !defined(TILE_K)
#error "WG, RB and TILE_K must be defined at build time"
#endif
#define TILE_M (WG * RB)
#define TILE_N (WG * RB)
/* Patch rows loaded per work-item per step, and their spacing. */
#define KSTEP (WG * WG / TILE_N)
#define NLOAD (TILE_K / KSTEP)

/* Epilogue flags, applied in this order. */
#define EP_SILU 1     /* v = v * sigmoid(v) */
#define EP_RESIDUAL 2 /* v = v + res */
#define EP_SHUFFLE2 4 /* Output channel m goes to plane m / 4 of a 2x
                       * upsampled output, sub-pixel ((m % 4) / 2, m % 2). */

#ifdef ACT_HALF
typedef half act_t;
#define LD(p, i) vload_half((i), (p))
#define ST(p, i, v) vstore_half_rte((v), (i), (p))
#else
typedef float act_t;
#define LD(p, i) ((p)[i])
#define ST(p, i, v) ((p)[i] = (v))
#endif

/* Head dimension limit of the attention kernel (YOLO11 uses 64). */
#define MAX_HEAD_DIM 64

/* Convolution with kernel k x k, stride s, zero padding pad and one group,
 * computed as an implicit matrix product
 *   out[m][pixel] = sum_r wt[r][m] * patch[r][pixel] + bias[m],
 * r = ci * k * k + ky * k + kx, the patch values being gathered from the
 * input planes on the fly, followed by the epilogue of flags.
 *
 * in: cin planes of hin x win. wt: (kpad, mpad), kpad = cin * k * k
 * rounded up to TILE_K and mpad = cout rounded up to TILE_M, zero-filled;
 * bias: mpad values. out and res: cout planes of hout x wout (or, with
 * EP_SHUFFLE2, cout / 4 planes of 2 hout x 2 wout).
 * Global size (ceil(hout * wout / TILE_N) * WG, mpad / TILE_M * WG, batch),
 * local size (WG, WG, 1). */
__kernel __attribute__((reqd_work_group_size(WG, WG, 1))) void
conv_gemm(__global const act_t *in, int in_off, int in_bs, int cin, int hin,
          int win, __global const float *wt, __global const float *bias,
          __global act_t *out, int out_off, int out_bs, int cout, int hout,
          int wout, int k, int s, int pad, int flags,
          __global const act_t *res, int res_off, int res_bs)
{
    __local float ws[TILE_K][TILE_M];
    __local float xs[TILE_K][TILE_N];
    int tx = get_local_id(0), ty = get_local_id(1);
    int lid = ty * WG + tx, b = get_global_id(2);
    int n0 = get_group_id(0) * TILE_N, m0 = get_group_id(1) * TILE_M;
    int plane_in = hin * win, plane = hout * wout, kk = k * k;
    int ktot = cin * kk;
    int kpad = (ktot + TILE_K - 1) / TILE_K * TILE_K;
    int mpad = get_num_groups(1) * TILE_M;
    /* Each work-item loads one column of the patch block, which fixes the
     * output pixel it gathers for. */
    int ln = lid % TILE_N, lk = lid / TILE_N;
    int pn = n0 + ln, oy = pn / wout, ox = pn % wout;
    int iy0 = oy * s - pad, ix0 = ox * s - pad;
    __global const act_t *src = in + in_off + (long)b * in_bs;
    float acc[RB][RB];
    int i, j, k0, kq, l;

    for (i = 0; i < RB; i++)
        for (j = 0; j < RB; j++)
            acc[i][j] = 0.0f;

    for (k0 = 0; k0 < kpad; k0 += TILE_K) {
        for (l = 0; l < NLOAD; l++) {
            int kr = lk + l * KSTEP, r = k0 + kr;
            float v = 0.0f;

            ws[kr][ln] = wt[r * mpad + m0 + ln];
            if (r < ktot && pn < plane) {
                int ci = r / kk, t = r - ci * kk, ky = t / k;
                int yy = iy0 + ky, xx = ix0 + t - ky * k;

                if (yy >= 0 && yy < hin && xx >= 0 && xx < win)
                    v = LD(src, ci * plane_in + yy * win + xx);
            }
            xs[kr][ln] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (kq = 0; kq < TILE_K; kq++) {
            float a[RB], c[RB];

            for (i = 0; i < RB; i++)
                a[i] = ws[kq][ty + i * WG];
            for (j = 0; j < RB; j++)
                c[j] = xs[kq][tx + j * WG];
            for (i = 0; i < RB; i++)
                for (j = 0; j < RB; j++)
                    acc[i][j] = mad(a[i], c[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (j = 0; j < RB; j++) {
        int n = n0 + tx + j * WG;

        if (n >= plane)
            continue;
        for (i = 0; i < RB; i++) {
            int m = m0 + ty + i * WG;
            float v;
            long o;

            if (m >= cout)
                continue;
            v = acc[i][j] + bias[m];
            if (flags & EP_SILU)
                v = v / (1.0f + exp(-v));
            if (flags & EP_SHUFFLE2) {
                int y = n / wout, x = n - y * wout, sub = m & 3;

                o = (long)(m >> 2) * 4 * plane +
                    (2 * y + (sub >> 1)) * 2 * wout + 2 * x + (sub & 1);
            }
            else
                o = (long)m * plane + n;
            if (flags & EP_RESIDUAL)
                v += LD(res, res_off + (long)b * res_bs + o);
            ST(out, out_off + (long)b * out_bs + o, v);
        }
    }
}

/* Depthwise 3 x 3 convolution, stride 1, zero padding 1, then the
 * epilogue of flags (EP_SILU, EP_RESIDUAL). Output channel c reads input
 * channel (c / grp) * gstride + goff + c % grp, which with grp = c lets
 * the attention read its value channels straight from the qkv tensor.
 * wt: (c, 9), bias: c. Global size (w, h, c * batch). */
__kernel void dwconv3x3(__global const act_t *in, int in_off, int in_bs,
                        int grp, int gstride, int goff,
                        __global const float *wt, __global const float *bias,
                        __global act_t *out, int out_off, int out_bs, int c,
                        int h, int w, int flags, __global const act_t *res,
                        int res_off, int res_bs)
{
    int x = get_global_id(0), y = get_global_id(1);
    int ch = get_global_id(2) % c, b = get_global_id(2) / c;
    int plane = h * w, dy, dx;
    int ci = (ch / grp) * gstride + goff + ch % grp;
    __global const act_t *src = in + in_off + (long)b * in_bs + (long)ci * plane;
    __global const float *k = wt + ch * 9;
    float v = bias[ch];
    long o;

    if (x >= w || y >= h)
        return;
    for (dy = -1; dy <= 1; dy++) {
        int yy = y + dy;

        if (yy < 0 || yy >= h)
            continue;
        for (dx = -1; dx <= 1; dx++) {
            int xx = x + dx;

            if (xx >= 0 && xx < w)
                v = mad(k[(dy + 1) * 3 + dx + 1], LD(src, yy * w + xx), v);
        }
    }
    if (flags & EP_SILU)
        v = v / (1.0f + exp(-v));
    o = (long)ch * plane + y * w + x;
    if (flags & EP_RESIDUAL)
        v += LD(res, res_off + (long)b * res_bs + o);
    ST(out, out_off + (long)b * out_bs + o, v);
}

/* Max pooling with a k x k window, stride 1, padding k / 2 that never
 * wins (as torch's -inf padding). Global size (w, h, c * batch). */
__kernel void maxpool(__global const act_t *in, int in_off, int in_bs,
                      __global act_t *out, int out_off, int out_bs, int c,
                      int h, int w, int k)
{
    int x = get_global_id(0), y = get_global_id(1);
    int ch = get_global_id(2) % c, b = get_global_id(2) / c;
    int plane = h * w, r = k / 2, dy, dx;
    __global const act_t *src = in + in_off + (long)b * in_bs + (long)ch * plane;
    float v = -INFINITY;

    if (x >= w || y >= h)
        return;
    for (dy = -r; dy <= r; dy++) {
        int yy = y + dy;

        if (yy < 0 || yy >= h)
            continue;
        for (dx = -r; dx <= r; dx++) {
            int xx = x + dx;

            if (xx >= 0 && xx < w)
                v = fmax(v, LD(src, yy * w + xx));
        }
    }
    ST(out, out_off + (long)b * out_bs + (long)ch * plane + y * w + x, v);
}

/* Nearest-neighbour 2x upsampling of h x w planes. Global size
 * (2 w, 2 h, c * batch). */
__kernel void upsample2(__global const act_t *in, int in_off, int in_bs,
                        __global act_t *out, int out_off, int out_bs, int c,
                        int h, int w)
{
    int x = get_global_id(0), y = get_global_id(1);
    int ch = get_global_id(2) % c, b = get_global_id(2) / c;

    if (x >= 2 * w || y >= 2 * h)
        return;
    ST(out, out_off + (long)b * out_bs + ((long)ch * 2 * h + y) * 2 * w + x,
       LD(in, in_off + (long)b * in_bs + ((long)ch * h + y / 2) * w + x / 2));
}

/* Copy n = c * h * w contiguous values per image, for concatenations.
 * Global size (n, batch). */
__kernel void copy_planes(__global const act_t *in, int in_off, int in_bs,
                          __global act_t *out, int out_off, int out_bs, int n)
{
    int i = get_global_id(0), b = get_global_id(1);

    if (i < n)
        ST(out, out_off + (long)b * out_bs + i,
           LD(in, in_off + (long)b * in_bs + i));
}

/* Multi-head self-attention of the PSA block over the n pixels of a
 * plane. For head h the qkv tensor holds kd query, kd key and hd value
 * channels, starting at channel h * (2 kd + hd). Output channel
 * h * hd + e of pixel i is
 *   sum_j v[e][j] softmax_j(scale * sum_d q[d][i] k[d][j]),
 * computed with an online softmax. Global size (n, heads, batch). */
__kernel void attention(__global const act_t *qkv, int qkv_off, int qkv_bs,
                        __global act_t *out, int out_off, int out_bs, int n,
                        int kd, int hd, float scale)
{
    int i = get_global_id(0), h = get_global_id(1), b = get_global_id(2);
    __global const act_t *q =
        qkv + qkv_off + (long)b * qkv_bs + (long)h * (2 * kd + hd) * n;
    __global const act_t *kt = q + (long)kd * n;
    __global const act_t *v = q + (long)2 * kd * n;
    float acc[MAX_HEAD_DIM], mx = -INFINITY, sum = 0.0f;
    int j, d, e;

    if (i >= n)
        return;
    for (e = 0; e < hd; e++)
        acc[e] = 0.0f;
    for (j = 0; j < n; j++) {
        float sc = 0.0f, p, corr;

        for (d = 0; d < kd; d++)
            sc = mad(LD(q, d * n + i), LD(kt, d * n + j), sc);
        sc *= scale;
        if (sc > mx) {
            corr = exp(mx - sc);
            sum *= corr;
            for (e = 0; e < hd; e++)
                acc[e] *= corr;
            mx = sc;
        }
        p = exp(sc - mx);
        sum += p;
        for (e = 0; e < hd; e++)
            acc[e] = mad(p, LD(v, e * n + j), acc[e]);
    }
    for (e = 0; e < hd; e++)
        ST(out, out_off + (long)b * out_bs + (long)(h * hd + e) * n + i,
           acc[e] / sum);
}
