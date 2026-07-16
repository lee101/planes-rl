#include "common.h"
#include "sim.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

#define RW 1280
#define RH 720
#define FPS 24

struct Cam { v3 pos, right, up, fwd; float tx, ty; };
struct RTri { float ax, ay, az, bx, by, bz, cx, cy, cz; uint32_t col; };

static Cam make_cam(v3 pos, v3 target, float fov_deg) {
    Cam c; c.pos = pos;
    c.fwd = norm3(sub(target, pos));
    c.right = norm3(cross(c.fwd, V3(0, 0, 1)));
    c.up = cross(c.right, c.fwd);
    c.ty = tanf(fov_deg * 0.5f * 3.14159265358979323846f / 180.f);
    c.tx = c.ty * (float)RW / RH;
    return c;
}

__device__ __forceinline__ uint32_t pack_rgb(float r, float g, float b) {
    return ((uint32_t)(fminf(fmaxf(r, 0.f), 1.f) * 255) << 16) |
           ((uint32_t)(fminf(fmaxf(g, 0.f), 1.f) * 255) << 8) |
           (uint32_t)(fminf(fmaxf(b, 0.f), 1.f) * 255) | 0xff000000u;
}

__device__ __forceinline__ void sky_ground(Cam cam, float px, float py, uint32_t* col, float* depth) {
    float xn = (2.f * px / RW - 1.f) * cam.tx;
    float yn = (1.f - 2.f * py / RH) * cam.ty;
    v3 dir = norm3(add(add(scl(cam.right, xn), scl(cam.up, yn)), cam.fwd));
    float skyt = fmaxf(0.f, dir.z);
    float sr = 0.62f - 0.25f * skyt, sg = 0.76f - 0.22f * skyt, sb = 0.92f - 0.10f * skyt;
    if (dir.z < -1e-5f && cam.pos.z > 0) {
        float t = -cam.pos.z / dir.z;
        v3 h = add(cam.pos, scl(dir, t));
        int cx = (int)floorf(h.x), cy = (int)floorf(h.y);
        float ck = ((cx + cy) & 1) ? 0.86f : 0.78f;
        float r = 0.55f * ck, g = 0.68f * ck, b = 0.48f * ck;
        float fx = h.x / 5.f;
        if (fabsf(fx - roundf(fx)) < 0.012f && h.x > 0.5f) { r = g = b = 0.95f; } // 5m distance lines
        if (fabsf(h.x) < 0.03f || fabsf(h.y) < 0.02f) { r = 0.9f; g = 0.4f; b = 0.3f; } // origin axes
        float fog = expf(-t / 140.f);
        r = r * fog + sr * (1 - fog); g = g * fog + sg * (1 - fog); b = b * fog + sb * (1 - fog);
        *col = pack_rgb(r, g, b);
        *depth = dot(dir, cam.fwd) * t;
    } else {
        *col = pack_rgb(sr, sg, sb);
        *depth = 1e30f;
    }
}

__global__ void k_clear(unsigned long long* zb, Cam cam) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= RW * RH) return;
    uint32_t col; float d;
    sky_ground(cam, (i % RW) + 0.5f, (i / RW) + 0.5f, &col, &d);
    zb[i] = ((unsigned long long)__float_as_uint(d) << 32) | col;
}

__global__ void k_raster(const RTri* tris, int n, unsigned long long* zb, Cam cam) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    RTri t = tris[i];
    if (t.col == 0) return;
    v3 P[3] = { V3(t.ax, t.ay, t.az), V3(t.bx, t.by, t.bz), V3(t.cx, t.cy, t.cz) };
    float sx[3], sy[3], sz[3];
    for (int k = 0; k < 3; k++) {
        v3 d = sub(P[k], cam.pos);
        float z = dot(d, cam.fwd);
        if (z < 0.03f) return;
        sx[k] = (dot(d, cam.right) / (z * cam.tx) * 0.5f + 0.5f) * RW;
        sy[k] = (0.5f - dot(d, cam.up) / (z * cam.ty) * 0.5f) * RH;
        sz[k] = z;
    }
    int x0 = max(0, (int)floorf(fminf(sx[0], fminf(sx[1], sx[2]))));
    int x1 = min(RW - 1, (int)ceilf(fmaxf(sx[0], fmaxf(sx[1], sx[2]))));
    int y0 = max(0, (int)floorf(fminf(sy[0], fminf(sy[1], sy[2]))));
    int y1 = min(RH - 1, (int)ceilf(fmaxf(sy[0], fmaxf(sy[1], sy[2]))));
    if (x1 < x0 || y1 < y0) return;
    if ((x1 - x0) * (y1 - y0) > 400000) return; // sanity cap
    float d12x = sx[1] - sx[0], d12y = sy[1] - sy[0];
    float d13x = sx[2] - sx[0], d13y = sy[2] - sy[0];
    float det = d12x * d13y - d13x * d12y;
    if (fabsf(det) < 1e-9f) return;
    float idet = 1.f / det;
    float iz0 = 1.f / sz[0], iz1 = 1.f / sz[1], iz2 = 1.f / sz[2];
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++) {
            float ex = x + 0.5f - sx[0], ey = y + 0.5f - sy[0];
            float b1 = (ex * d13y - d13x * ey) * idet;
            float b2 = (d12x * ey - ex * d12y) * idet;
            if (b1 < 0 || b2 < 0 || b1 + b2 > 1) continue;
            float iz = iz0 * (1 - b1 - b2) + iz1 * b1 + iz2 * b2;
            float z = 1.f / iz;
            unsigned long long key = ((unsigned long long)__float_as_uint(z) << 32) | t.col;
            atomicMin(&zb[y * RW + x], key);
        }
}

__global__ void k_resolve(const unsigned long long* zb, uint8_t* rgb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= RW * RH) return;
    uint32_t c = (uint32_t)zb[i];
    rgb[i * 3] = (c >> 16) & 0xff; rgb[i * 3 + 1] = (c >> 8) & 0xff; rgb[i * 3 + 2] = c & 0xff;
}

__global__ void k_progress(uint8_t* rgb, int frame, int frames) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= RW * RH) return;
    int x = i % RW, y = i / RW;
    const int x0 = 28, x1 = RW - 28, y0 = RH - 27, y1 = RH - 15;
    if (x < x0 || x >= x1 || y < y0 || y >= y1) return;
    float p = (float)(frame + 1) / frames;
    int fill = x0 + (int)((x1 - x0) * p);
    bool tick = ((x - x0) % ((x1 - x0) / 10) < 2);
    uint8_t r = x <= fill ? 78 : 18, g = x <= fill ? 210 : 24, b = x <= fill ? 255 : 30;
    if (tick) r = g = b = 235;
    rgb[i*3] = r; rgb[i*3+1] = g; rgb[i*3+2] = b;
}

// ---- mesh generation for rendering (COM-centered body frame) ----
// one block per plane; shared-memory mesh (avoids huge per-thread local reservation)
__global__ void k_meshgen(const Genome* gs, int n, v3* verts, int* tris, int* nvs, int* nts) {
    int i = blockIdx.x;
    if (i >= n) return;
    __shared__ v3 V[MAX_V]; __shared__ int T[MAX_T][3];
    __shared__ MassProps s_mp; __shared__ MeshOut s_mo;
    if (threadIdx.x == 0) {
        MeshOut mo;
        Genome g = gs[i];
        build_mesh(&g, V, T, &mo);
        MassProps mp; mass_props(&g, V, T, mo.nt, &mp);
        s_mp = mp; s_mo = mo;
        nvs[i] = mo.nv; nts[i] = mo.nt;
    }
    __syncthreads();
    for (int v = threadIdx.x; v < s_mo.nv; v += blockDim.x) verts[i * MAX_V + v] = sub(V[v], s_mp.com);
    for (int t = threadIdx.x; t < s_mo.nt; t += blockDim.x) {
        tris[(i * MAX_T + t) * 3] = T[t][0];
        tris[(i * MAX_T + t) * 3 + 1] = T[t][1];
        tris[(i * MAX_T + t) * 3 + 2] = T[t][2];
    }
}

__device__ __forceinline__ v3 pose_xform(float8 r, v3 p) {
    quat q = Q(r.d, r.e, r.f, r.g);
    return add(qrot(q, p), V3(r.a, r.b, r.c));
}

__device__ uint32_t plane_color(int p, int n, float shade) {
    float h = fmodf(0.62f + 2.4f * (float)p / n, 1.0f) * 6.f;
    int hi = (int)h; float f = h - hi;
    float r, g, b, v = 0.95f, s = 0.75f;
    float pp = v * (1 - s), qq = v * (1 - s * f), tt = v * (1 - s * (1 - f));
    switch (hi % 6) {
        case 0: r = v; g = tt; b = pp; break; case 1: r = qq; g = v; b = pp; break;
        case 2: r = pp; g = v; b = tt; break; case 3: r = pp; g = qq; b = v; break;
        case 4: r = tt; g = pp; b = v; break; default: r = v; g = pp; b = qq; break;
    }
    return pack_rgb(r * shade, g * shade, b * shade);
}

// emit shaded world-space triangles for all planes at frame f
__global__ void k_emit(const v3* verts, const int* tris, const int* nts,
                       const float8* rec, int rec_cap, int frame, int nplanes,
                       RTri* out, int shadow) {
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    int p = gi / MAX_T, t = gi % MAX_T;
    if (p >= nplanes) return;
    RTri o; o.col = 0;
    if (t < nts[p]) {
        float8 r = rec[p * rec_cap + frame];
        v3 a = pose_xform(r, verts[p * MAX_V + tris[(p * MAX_T + t) * 3]]);
        v3 b = pose_xform(r, verts[p * MAX_V + tris[(p * MAX_T + t) * 3 + 1]]);
        v3 c = pose_xform(r, verts[p * MAX_V + tris[(p * MAX_T + t) * 3 + 2]]);
        if (shadow) {
            float k = fmaxf(0.f, 1.f - (a.z) * 0.9f);
            o.col = pack_rgb(0.38f * k + 0.5f * (1 - k), 0.47f * k + 0.6f * (1 - k), 0.34f * k + 0.45f * (1 - k));
            a.z = b.z = c.z = 0.012f;
        } else {
            v3 n = norm3(cross(sub(b, a), sub(c, a)));
            v3 L = norm3(V3(0.35f, 0.25f, 0.9f));
            float sh = 0.35f + 0.65f * fabsf(dot(n, L));
            o.col = plane_color(p, nplanes, sh);
        }
        o.ax = a.x; o.ay = a.y; o.az = a.z; o.bx = b.x; o.by = b.y; o.bz = b.z; o.cx = c.x; o.cy = c.y; o.cz = c.z;
    }
    out[gi] = o;
}

// Compact emitter: one block per plane and exact prefix-sum output ranges.  The
// old flat MAX_T launch is retained for the one-plane turntable, while swarm
// rendering avoids rasterizing hundreds of thousands of empty triangle slots.
__global__ void k_emit_compact(const v3* verts, const int* tris, const int* nts,
                               const int* offsets, const float8* rec, int rec_cap,
                               int frame, int nplanes, RTri* out, int shadow) {
    int p = blockIdx.x;
    if (p >= nplanes) return;
    float8 r = rec[p * rec_cap + frame];
    for (int t = threadIdx.x; t < nts[p]; t += blockDim.x) {
        RTri o;
        v3 a = pose_xform(r, verts[p * MAX_V + tris[(p * MAX_T + t) * 3]]);
        v3 b = pose_xform(r, verts[p * MAX_V + tris[(p * MAX_T + t) * 3 + 1]]);
        v3 c = pose_xform(r, verts[p * MAX_V + tris[(p * MAX_T + t) * 3 + 2]]);
        if (shadow) {
            float k = fmaxf(0.f, 1.f - a.z * 0.9f);
            o.col = pack_rgb(0.38f * k + 0.5f * (1-k), 0.47f * k + 0.6f * (1-k), 0.34f * k + 0.45f * (1-k));
            a.z = b.z = c.z = 0.012f;
        } else {
            v3 n = norm3(cross(sub(b, a), sub(c, a)));
            v3 L = norm3(V3(0.35f, 0.25f, 0.9f));
            o.col = plane_color(p, nplanes, 0.35f + 0.65f * fabsf(dot(n, L)));
        }
        o.ax=a.x; o.ay=a.y; o.az=a.z; o.bx=b.x; o.by=b.y; o.bz=b.z; o.cx=c.x; o.cy=c.y; o.cz=c.z;
        out[offsets[p] + t] = o;
    }
}

#define RIBBON 240
#define TRAIL_STRIDE 3
__global__ void k_emit_ribbons(const float8* rec, int rec_cap, int frame, int nplanes, Cam cam, RTri* out) {
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    int p = gi / RIBBON, s = gi % RIBBON;
    if (p >= nplanes) return;
    RTri o1; o1.col = 0; RTri o2; o2.col = 0;
    int f1 = frame - s * TRAIL_STRIDE, f0 = f1 - TRAIL_STRIDE;
    if (f0 >= 0 && f1 <= frame) {
        float8 ra = rec[p * rec_cap + f0], rb = rec[p * rec_cap + f1];
        if (rb.h > 0.5f) { // only while airborne
            v3 a = V3(ra.a, ra.b, ra.c), b = V3(rb.a, rb.b, rb.c);
            v3 d = sub(b, a);
            if (dot(d, d) > 1e-8f) {
                v3 view = norm3(sub(scl(add(a, b), 0.5f), cam.pos));
                v3 w = norm3(cross(d, view));
                float hw = 0.006f * (1.f - (float)s / RIBBON);
                v3 a0 = add(a, scl(w, hw)), a1 = sub(a, scl(w, hw));
                v3 b0 = add(b, scl(w, hw)), b1 = sub(b, scl(w, hw));
                float fade = 1.f - (float)s / RIBBON;
                uint32_t base = plane_color(p, nplanes, 0.55f + 0.35f * fade);
                o1.col = base; o2.col = base;
                o1.ax = a0.x; o1.ay = a0.y; o1.az = a0.z; o1.bx = b0.x; o1.by = b0.y; o1.bz = b0.z; o1.cx = a1.x; o1.cy = a1.y; o1.cz = a1.z;
                o2.ax = b0.x; o2.ay = b0.y; o2.az = b0.z; o2.bx = b1.x; o2.by = b1.y; o2.bz = b1.z; o2.cx = a1.x; o2.cy = a1.y; o2.cz = a1.z;
            }
        }
    }
    out[gi * 2] = o1; out[gi * 2 + 1] = o2;
}

#define NPART 6000
__global__ void k_part_init(v3* pp, v3 center) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NPART) return;
    float a = i * 0.6180339887f, b = i * 0.7548776662f, c = i * 0.5698402910f;
    pp[i] = V3(center.x - 12.f + 60.f * (a - floorf(a)), center.y - 25.f + 50.f * (b - floorf(b)), 0.2f + 7.f * (c - floorf(c)));
}
__global__ void k_part_step(v3* pp, float t, int seed, v3 center) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NPART) return;
    v3 p = pp[i];
    v3 w = wind_at(p, t, seed);
    p = add(p, scl(w, 1.0f / FPS));
    if (p.x < center.x - 14.f || p.x > center.x + 50.f || fabsf(p.y - center.y) > 26.f || p.z < 0.05f || p.z > 8.f) {
        float a = (i * 12.9898f + t * 3.7f); a -= floorf(a);
        float b = (i * 78.233f + t * 1.3f); b -= floorf(b);
        float c = (i * 39.425f + t * 2.1f); c -= floorf(c);
        p = V3(center.x - 12.f + 60.f * a, center.y - 25.f + 50.f * b, 0.2f + 7.f * c);
    }
    pp[i] = p;
}
__global__ void k_emit_parts(const v3* pp, float t, int seed, Cam cam, RTri* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NPART) return;
    RTri o; o.col = 0;
    v3 p = pp[i];
    v3 w = wind_at(p, t, seed);
    v3 d = scl(w, 0.09f);
    v3 view = norm3(sub(p, cam.pos));
    v3 side = scl(norm3(cross(d, view)), 0.0035f);
    v3 a = sub(p, d), b = add(p, d);
    o.col = pack_rgb(0.97f, 0.97f, 1.0f);
    o.ax = a.x; o.ay = a.y; o.az = a.z;
    o.bx = b.x + side.x; o.by = b.y + side.y; o.bz = b.z + side.z;
    o.cx = b.x - side.x; o.cy = b.y - side.y; o.cz = b.z - side.z;
    out[i] = o;
}

// ---------------- host drivers ----------------
static FILE* open_ffmpeg(const char* path, const char* label) {
    char cmd[1400];
#ifdef _WIN32
    // Scoop ffmpeg builds may not ship a usable fontconfig setup. Keep the
    // Windows pipe dependency-free; the simulation itself supplies the visual
    // wind/path story and the filename/console retain scenario statistics.
    (void)label;
    snprintf(cmd, sizeof cmd,
             "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - "
             "-c:v libx264 -pix_fmt yuv420p -crf 22 -preset slow -maxrate 8M -bufsize 16M -movflags +faststart \"%s\"",
             RW, RH, FPS, path);
#else
    snprintf(cmd, sizeof cmd,
             "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - "
             "-vf \"drawtext=text='%s':x=28:y=24:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.35:boxborderw=10\" "
             "-c:v libx264 -pix_fmt yuv420p -crf 22 -preset slow -maxrate 8M -bufsize 16M -movflags +faststart \"%s\"",
             RW, RH, FPS, label, path);
#endif
#ifdef _WIN32
    FILE* f = popen(cmd, "wb");
#else
    FILE* f = popen(cmd, "w");
#endif
    if (!f) { fprintf(stderr, "ffmpeg spawn failed\n"); exit(1); }
    return f;
}

struct Recorded {
    int n, rec_cap;
    float8* d_rec;
    v3* d_verts; int* d_tris; int* d_nts; int* d_nvs; int* d_offsets;
    int tri_count;
    std::vector<float8> h_rec;
    std::vector<float> fitness;
};

static Recorded record_flights(const Genome* genomes, int n, int seed, float spread, int seconds, int vary_scenarios) {
    Recorded R; R.n = n; R.rec_cap = seconds * FPS;
    Genome* d_gs; CK(cudaMalloc(&d_gs, sizeof(Genome) * n));
    CK(cudaMemcpy(d_gs, genomes, sizeof(Genome) * n, cudaMemcpyHostToDevice));
    void* pb = panelbuf_alloc(n);
    gpu_prepare(d_gs, pb, n);
    CK(cudaMalloc(&R.d_rec, sizeof(float8) * n * R.rec_cap));
    CK(cudaMemset(R.d_rec, 0, sizeof(float8) * n * R.rec_cap));
    float* d_fit; CK(cudaMalloc(&d_fit, sizeof(float) * n));
    int rec_every = (int)(1.0f / (SIM_DT * FPS) + 0.5f); // 25 at 24 fps
    gpu_sim(pb, n, seed, d_fit, R.d_rec, rec_every, R.rec_cap, spread, vary_scenarios);
    R.fitness.resize(n);
    CK(cudaMemcpy(R.fitness.data(), d_fit, sizeof(float) * n, cudaMemcpyDeviceToHost));
    CK(cudaMalloc(&R.d_verts, sizeof(v3) * n * MAX_V));
    CK(cudaMalloc(&R.d_tris, sizeof(int) * n * MAX_T * 3));
    CK(cudaMalloc(&R.d_nts, sizeof(int) * n));
    CK(cudaMalloc(&R.d_nvs, sizeof(int) * n));
    k_meshgen<<<n, 64>>>(d_gs, n, R.d_verts, R.d_tris, R.d_nvs, R.d_nts);
    CK(cudaDeviceSynchronize());
    std::vector<int> h_nts(n), offsets(n + 1, 0);
    CK(cudaMemcpy(h_nts.data(), R.d_nts, sizeof(int) * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) offsets[i + 1] = offsets[i] + h_nts[i];
    R.tri_count = offsets[n];
    CK(cudaMalloc(&R.d_offsets, sizeof(int) * (n + 1)));
    CK(cudaMemcpy(R.d_offsets, offsets.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    R.h_rec.resize((size_t)n * R.rec_cap);
    CK(cudaMemcpy(R.h_rec.data(), R.d_rec, sizeof(float8) * n * R.rec_cap, cudaMemcpyDeviceToHost));
    panelbuf_free(pb); CK(cudaFree(d_fit)); CK(cudaFree(d_gs));
    return R;
}

static void render_frames(Recorded& R, int frames, const char* out, const char* label,
                          int mode /*0=showcase,1=hero*/, int wind_seed) {
    int cap_planes = R.tri_count;
    int cap_rib = R.n * RIBBON * 2;
    int total = cap_planes * 2 + cap_rib + NPART;
    RTri* d_tris; CK(cudaMalloc(&d_tris, sizeof(RTri) * total));
    unsigned long long* d_zb; CK(cudaMalloc(&d_zb, sizeof(unsigned long long) * RW * RH));
    uint8_t* d_rgb; CK(cudaMalloc(&d_rgb, RW * RH * 3));
    uint8_t* h_rgb; CK(cudaHostAlloc(&h_rgb, RW * RH * 3, cudaHostAllocDefault));
    v3* d_pp; CK(cudaMalloc(&d_pp, sizeof(v3) * NPART));
    FILE* ff = open_ffmpeg(out, label);
    v3 cam_target = V3(4, 0, 1.5f); v3 prev_target = cam_target;
    k_part_init<<<(NPART + 127) / 128, 128>>>(d_pp, V3(10, 0, 2));
    for (int f = 0; f < frames; f++) {
        // centroid of airborne planes from host rec
        v3 cen = V3(0, 0, 0); float wsum = 0; float maxx = 0;
        for (int p = 0; p < R.n; p++) {
            float8 r = R.h_rec[(size_t)p * R.rec_cap + f];
            float w = r.h > 0.5f ? 1.f : 0.08f;
            cen = add(cen, scl(V3(r.a, r.b, r.c), w)); wsum += w;
            maxx = fmaxf(maxx, r.a);
        }
        cen = scl(cen, 1.f / fmaxf(wsum, 1e-3f));
        cam_target = add(scl(prev_target, 0.94f), scl(V3(cen.x, cen.y, fmaxf(cen.z, 0.6f)), 0.06f));
        prev_target = cam_target;
        Cam cam;
        float tt = (float)f / frames;
        if (mode == 0) {
            float ang = -2.1f + 1.15f * tt;
            float rad = 13.f + 9.f * tt;
            v3 cp = add(cam_target, V3(rad * cosf(ang), rad * sinf(ang), 3.2f + 2.5f * tt));
            cam = make_cam(cp, cam_target, 55);
        } else {
            float8 r = R.h_rec[f];
            v3 pp = V3(r.a, r.b, r.c);
            v3 cp = add(pp, V3(-1.6f, 0.9f - 1.4f * tt, 0.45f));
            cam = make_cam(cp, add(pp, V3(0.8f, 0, 0)), 58);
        }
        float simt = (float)f / FPS;
        k_clear<<<(RW * RH + 255) / 256, 256>>>(d_zb, cam);
        k_emit_compact<<<R.n, 128>>>(R.d_verts, R.d_tris, R.d_nts, R.d_offsets, R.d_rec, R.rec_cap, f, R.n, d_tris, 0);
        k_emit_compact<<<R.n, 128>>>(R.d_verts, R.d_tris, R.d_nts, R.d_offsets, R.d_rec, R.rec_cap, f, R.n, d_tris + cap_planes, 1);
        k_emit_ribbons<<<(R.n * RIBBON + 127) / 128, 128>>>(R.d_rec, R.rec_cap, f, R.n, cam, d_tris + cap_planes * 2);
        k_part_step<<<(NPART + 127) / 128, 128>>>(d_pp, simt, wind_seed, cam_target);
        k_emit_parts<<<(NPART + 127) / 128, 128>>>(d_pp, simt, wind_seed, cam, d_tris + cap_planes * 2 + cap_rib);
        k_raster<<<(total + 127) / 128, 128>>>(d_tris, total, d_zb, cam);
        k_resolve<<<(RW * RH + 255) / 256, 256>>>(d_zb, d_rgb);
        CK(cudaMemcpy(h_rgb, d_rgb, RW * RH * 3, cudaMemcpyDeviceToHost));
        fwrite(h_rgb, 1, RW * RH * 3, ff);
        if (f % 120 == 0) { printf("frame %d/%d\n", f, frames); fflush(stdout); }
    }
    pclose(ff);
    CK(cudaFree(d_tris)); CK(cudaFree(d_zb)); CK(cudaFree(d_rgb)); CK(cudaFreeHost(h_rgb)); CK(cudaFree(d_pp));
    CK(cudaFree(R.d_offsets)); CK(cudaFree(R.d_nts)); CK(cudaFree(R.d_nvs));
    CK(cudaFree(R.d_tris)); CK(cudaFree(R.d_verts)); CK(cudaFree(R.d_rec));
}

void render_showcase(const Genome* genomes, int n, const char* out, int seconds, const char* label) {
    Recorded R = record_flights(genomes, n, 4242, 0.55f, seconds, 1);
    float best = 0; for (float f : R.fitness) best = fmaxf(best, f);
    printf("showcase: %d planes, best in scene %.1fm\n", n, best);
    render_frames(R, seconds * FPS, out, label, 0, 4242);
}

void render_hero(const Genome* g, const char* out, int seconds, const char* label) {
    Recorded R = record_flights(g, 1, 4242, 0, seconds, 0);
    printf("hero flight: %.1fm\n", R.fitness[0]);
    render_frames(R, seconds * FPS, out, label, 1, 4242);
}

void render_robust(const Genome* g, int scenarios, const char* out, int seconds, const char* label) {
    if (scenarios < 4) scenarios = 4;
    std::vector<Genome> copies(scenarios, *g);
    Recorded R = record_flights(copies.data(), scenarios, 9001, 0.16f, seconds, 1);
    std::vector<float> f = R.fitness;
    std::sort(f.begin(), f.end());
    float mean = 0; for (float x : f) mean += x / scenarios;
    printf("robust video: %d environments, min %.1fm median %.1fm mean %.1fm max %.1fm\n",
           scenarios, f.front(), f[f.size()/2], mean, f.back());
    render_frames(R, seconds * FPS, out, label, 0, 9001);
}

// Chronological geometry reel. Every archive champion is visited in order and
// adjacent genomes are interpolated, making the evolving wing/fin/tail planform
// readable instead of launching all generations as one indistinguishable swarm.
void render_evolution(const Genome* gs, int n, const char* out, int seconds, const char* label) {
    if (n < 2 || seconds < 1) { fprintf(stderr, "evolution-video needs >=2 genomes and >=1 second\n"); return; }
    Genome* d_g; CK(cudaMalloc(&d_g, sizeof(Genome)));
    v3* d_verts; int* d_tris; int* d_nts; int* d_nvs;
    CK(cudaMalloc(&d_verts, sizeof(v3) * MAX_V)); CK(cudaMalloc(&d_tris, sizeof(int) * MAX_T * 3));
    CK(cudaMalloc(&d_nts, sizeof(int))); CK(cudaMalloc(&d_nvs, sizeof(int)));
    int frames = seconds * FPS;
    float8* d_rec; CK(cudaMalloc(&d_rec, sizeof(float8) * frames));
    std::vector<float8> poses(frames);
    for (int f = 0; f < frames; f++) {
        float ang = 2.f * 3.14159265358979323846f * 1.35f * f / frames;
        quat q = qmul(qaxis(V3(0, 0, 1), ang), qaxis(V3(0, 1, 0), -0.12f));
        float8 r; r.a=0; r.b=0; r.c=0.16f; r.d=q.w; r.e=q.x; r.f=q.y; r.g=q.z; r.h=1; poses[f]=r;
    }
    CK(cudaMemcpy(d_rec, poses.data(), sizeof(float8) * frames, cudaMemcpyHostToDevice));
    RTri* d_rt; CK(cudaMalloc(&d_rt, sizeof(RTri) * MAX_T * 2));
    unsigned long long* d_zb; CK(cudaMalloc(&d_zb, sizeof(unsigned long long) * RW * RH));
    uint8_t* d_rgb; CK(cudaMalloc(&d_rgb, RW * RH * 3));
    uint8_t* h_rgb; CK(cudaHostAlloc(&h_rgb, RW * RH * 3, cudaHostAllocDefault));
    FILE* ff = open_ffmpeg(out, label);
    for (int f = 0; f < frames; f++) {
        float x = (float)f * (n - 1) / fmaxf(1.f, (float)(frames - 1));
        int a = (int)floorf(x), b = std::min(n - 1, a + 1); float u = x - a;
        Genome g;
        for (int j = 0; j < GENOME_N; j++)
            ((float*)&g)[j] = ((const float*)&gs[a])[j] * (1.f-u) + ((const float*)&gs[b])[j] * u;
        CK(cudaMemcpy(d_g, &g, sizeof g, cudaMemcpyHostToDevice));
        k_meshgen<<<1, 64>>>(d_g, 1, d_verts, d_tris, d_nvs, d_nts);
        Cam cam = make_cam(V3(0.53f, 0.0f, 0.33f), V3(-0.01f, 0, 0.13f), 43);
        k_clear<<<(RW * RH + 255) / 256, 256>>>(d_zb, cam);
        k_emit<<<(MAX_T + 127) / 128, 128>>>(d_verts, d_tris, d_nts, d_rec, frames, f, 1, d_rt + MAX_T, 1);
        k_emit<<<(MAX_T + 127) / 128, 128>>>(d_verts, d_tris, d_nts, d_rec, frames, f, 1, d_rt, 0);
        k_raster<<<(MAX_T * 2 + 127) / 128, 128>>>(d_rt, MAX_T * 2, d_zb, cam);
        k_resolve<<<(RW * RH + 255) / 256, 256>>>(d_zb, d_rgb);
        k_progress<<<(RW * RH + 255) / 256, 256>>>(d_rgb, f, frames);
        CK(cudaMemcpy(h_rgb, d_rgb, RW * RH * 3, cudaMemcpyDeviceToHost));
        fwrite(h_rgb, 1, RW * RH * 3, ff);
        if (f % (FPS * 2) == 0) { printf("evolution frame %d/%d generation %d/%d\n", f, frames, a + 1, n); fflush(stdout); }
    }
    pclose(ff);
    CK(cudaFreeHost(h_rgb)); CK(cudaFree(d_rgb)); CK(cudaFree(d_zb)); CK(cudaFree(d_rt));
    CK(cudaFree(d_rec)); CK(cudaFree(d_nts)); CK(cudaFree(d_nvs)); CK(cudaFree(d_tris));
    CK(cudaFree(d_verts)); CK(cudaFree(d_g));
}

// ---- turntable ----
void render_turntable(const Genome* g, const char* out, int seconds, const char* label) {
    Genome* d_g; CK(cudaMalloc(&d_g, sizeof(Genome)));
    CK(cudaMemcpy(d_g, g, sizeof(Genome), cudaMemcpyHostToDevice));
    v3* d_verts; int* d_tris; int* d_nts; int* d_nvs;
    CK(cudaMalloc(&d_verts, sizeof(v3) * MAX_V)); CK(cudaMalloc(&d_tris, sizeof(int) * MAX_T * 3));
    CK(cudaMalloc(&d_nts, 4)); CK(cudaMalloc(&d_nvs, 4));
    k_meshgen<<<1, 64>>>(d_g, 1, d_verts, d_tris, d_nvs, d_nts);
    CK(cudaDeviceSynchronize());
    int frames = seconds * FPS;
    float8* d_rec; CK(cudaMalloc(&d_rec, sizeof(float8) * frames));
    std::vector<float8> h_rec(frames);
    for (int f = 0; f < frames; f++) {
        float ang = 2.f * 3.14159265358979323846f * f / frames;
        quat q = qaxis(V3(0, 0, 1), ang);
        float8 r; r.a = 0; r.b = 0; r.c = 0.16f; r.d = q.w; r.e = q.x; r.f = q.y; r.g = q.z; r.h = 1;
        h_rec[f] = r;
    }
    CK(cudaMemcpy(d_rec, h_rec.data(), sizeof(float8) * frames, cudaMemcpyHostToDevice));
    int cap = MAX_T;
    RTri* d_rt; CK(cudaMalloc(&d_rt, sizeof(RTri) * cap * 2));
    unsigned long long* d_zb; CK(cudaMalloc(&d_zb, sizeof(unsigned long long) * RW * RH));
    uint8_t* d_rgb; CK(cudaMalloc(&d_rgb, RW * RH * 3));
    uint8_t* h_rgb; CK(cudaHostAlloc(&h_rgb, RW * RH * 3, cudaHostAllocDefault));
    FILE* ff = open_ffmpeg(out, label);
    for (int f = 0; f < frames; f++) {
        Cam cam = make_cam(V3(0.52f, 0.0f, 0.30f), V3(0, 0, 0.13f), 42);
        k_clear<<<(RW * RH + 255) / 256, 256>>>(d_zb, cam);
        k_emit<<<(cap + 127) / 128, 128>>>(d_verts, d_tris, d_nts, d_rec, frames, f, 1, d_rt + cap, 1); // shadow
        k_emit<<<(cap + 127) / 128, 128>>>(d_verts, d_tris, d_nts, d_rec, frames, f, 1, d_rt, 0);
        k_raster<<<(cap * 2 + 127) / 128, 128>>>(d_rt, cap * 2, d_zb, cam);
        k_resolve<<<(RW * RH + 255) / 256, 256>>>(d_zb, d_rgb);
        CK(cudaMemcpy(h_rgb, d_rgb, RW * RH * 3, cudaMemcpyDeviceToHost));
        fwrite(h_rgb, 1, RW * RH * 3, ff);
    }
    pclose(ff);
    CK(cudaFreeHost(h_rgb)); CK(cudaFree(d_rgb)); CK(cudaFree(d_zb)); CK(cudaFree(d_rt));
    CK(cudaFree(d_rec)); CK(cudaFree(d_nts)); CK(cudaFree(d_nvs)); CK(cudaFree(d_tris));
    CK(cudaFree(d_verts)); CK(cudaFree(d_g));
}
