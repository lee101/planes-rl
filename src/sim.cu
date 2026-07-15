#include "common.h"
#include "sim.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

// panels SoA per plane
struct PanelBuf {
    float4* pc;   // centroid.xyz, area   [pop*MAX_T]
    float4* pn;   // normal.xyz           [pop*MAX_T]
    MassProps* mp;
    int* nt;
    PlaneState* st;
};

// one block per plane; mesh built in shared memory (per-thread local arrays would
// reserve GBs of device local memory across all resident threads)
__global__ void k_prepare(const Genome* gs, PanelBuf pb, int pop) {
    int i = blockIdx.x;
    if (i >= pop) return;
    __shared__ v3 V[MAX_V]; __shared__ int T[MAX_T][3];
    __shared__ MassProps s_mp; __shared__ int s_nt;
    if (threadIdx.x == 0) {
        MeshOut mo;
        Genome g = gs[i];
        build_mesh(&g, V, T, &mo);
        MassProps mp; mass_props(&g, V, T, mo.nt, &mp);
        pb.mp[i] = mp; pb.nt[i] = mo.nt;
        s_mp = mp; s_nt = mo.nt;
    }
    __syncthreads();
    for (int t = threadIdx.x; t < s_nt; t += blockDim.x) {
        v3 a = sub(V[T[t][0]], s_mp.com), b = sub(V[T[t][1]], s_mp.com), c = sub(V[T[t][2]], s_mp.com);
        v3 nrm = cross(sub(b, a), sub(c, a));
        float A = 0.5f * len(nrm);
        v3 ct = scl(add(add(a, b), c), 1.0f / 3);
        pb.pc[i * MAX_T + t] = make_float4(ct.x, ct.y, ct.z, A);
        v3 n = A > 1e-12f ? scl(nrm, 0.5f / A) : V3(0, 0, 1);
        pb.pn[i * MAX_T + t] = make_float4(n.x, n.y, n.z, 0);
    }
}

#define BLK 128

// one block per plane; fused pose-transform + panel aero + reduce + integrate
// record: optional pose recording every rec_every steps into rec[plane*rec_cap + f] = {pos, quat, alive}
__global__ void k_sim(PanelBuf pb, int pop, int seed, float* fitness,
                      float8* rec, int rec_every, int rec_cap, float launch_spread,
                      PlaneState* st, int step0, int nsteps) {
    int p = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ v3 s_pos, s_vel, s_wb; __shared__ quat s_q;
    __shared__ float s_done, s_t;
    __shared__ float rF[3][BLK / 32], rT[3][BLK / 32];
    MassProps mp = pb.mp[p];
    int nt = pb.nt[p];
    if (tid == 0) {
        if (step0 == 0) {
            // staggered launch grid for showcase (spread=0 for GA)
            float ox = 0, oy = 0, oz = 0;
            if (launch_spread > 0) {
                int cols = 25;
                oy = ((p % cols) - cols / 2) * launch_spread;
                ox = -(p / cols) * launch_spread * 1.4f;
                oz = 0.25f * sinf(p * 12.9898f);
            }
            s_pos = V3(ox, oy, LAUNCH_H + oz);
            s_q = Q(1, 0, 0, 0);
            float jv = launch_spread > 0 ? 0.35f * sinf(p * 78.233f) : 0;
            s_vel = V3((LAUNCH_V + jv) * cosf(LAUNCH_PITCH), 0, (LAUNCH_V + jv) * sinf(LAUNCH_PITCH));
            s_wb = V3(0, 0, 0);
            s_done = 0; s_t = 0;
        } else {
            PlaneState s = st[p];
            s_pos = s.pos; s_q = s.q; s_vel = s.vel; s_wb = s.wb;
            s_done = s.done; s_t = step0 * SIM_DT;
        }
    }
    __syncthreads();
    int rec_i = step0 / rec_every;
    for (int step = step0; step < step0 + nsteps && step < SIM_MAX_STEPS; step++) {
        if (rec && step % rec_every == 0 && rec_i < rec_cap) {
            if (tid == 0) {
                float8 r; r.a = s_pos.x; r.b = s_pos.y; r.c = s_pos.z;
                r.d = s_q.w; r.e = s_q.x; r.f = s_q.y; r.g = s_q.z;
                r.h = s_done > 0 ? 0.0f : 1.0f;
                rec[p * rec_cap + rec_i] = r;
            }
            rec_i++;
        }
        if (s_done > 0) { if (!rec) break; else { __syncthreads(); continue; } }
        v3 r0, r1, r2; qmat(s_q, &r0, &r1, &r2);
        v3 vel = s_vel, wb = s_wb, pos = s_pos;
        v3 ww = V3(dot(r0, wb), dot(r1, wb), dot(r2, wb)); // omega world
        v3 wind = wind_at(pos, s_t, seed);
        v3 F = V3(0, 0, 0), Tq = V3(0, 0, 0);
        for (int t = tid; t < nt; t += BLK) {
            float4 c4 = pb.pc[p * MAX_T + t], n4 = pb.pn[p * MAX_T + t];
            v3 rb = V3(c4.x, c4.y, c4.z); float A = c4.w;
            v3 nb = V3(n4.x, n4.y, n4.z);
            v3 rw = V3(dot(r0, rb), dot(r1, rb), dot(r2, rb));
            v3 nw = V3(dot(r0, nb), dot(r1, nb), dot(r2, nb));
            v3 vr = sub(add(vel, cross(ww, rw)), wind);
            float V2 = dot(vr, vr);
            if (V2 < 1e-6f) continue;
            float Vm = sqrtf(V2);
            v3 vh = scl(vr, 1.0f / Vm);
            float s = dot(vh, nw);
            float qd = 0.5f * AIR_RHO * V2 * A;
            float cn = CN_SLOPE * s * sqrtf(fmaxf(0.f, 1.f - s * s)) + 2.0f * s * fabsf(s);
            v3 f = scl(nw, -qd * cn);
            v3 tang = sub(vh, scl(nw, s));
            f = sub(f, scl(tang, qd * CF_FRICTION));
            F = add(F, f);
            Tq = add(Tq, cross(rw, f));
        }
        // warp + block reduce
        for (int o = 16; o > 0; o >>= 1) {
            F.x += __shfl_down_sync(~0u, F.x, o); F.y += __shfl_down_sync(~0u, F.y, o); F.z += __shfl_down_sync(~0u, F.z, o);
            Tq.x += __shfl_down_sync(~0u, Tq.x, o); Tq.y += __shfl_down_sync(~0u, Tq.y, o); Tq.z += __shfl_down_sync(~0u, Tq.z, o);
        }
        if ((tid & 31) == 0) {
            int w = tid >> 5;
            rF[0][w] = F.x; rF[1][w] = F.y; rF[2][w] = F.z;
            rT[0][w] = Tq.x; rT[1][w] = Tq.y; rT[2][w] = Tq.z;
        }
        __syncthreads();
        if (tid == 0) {
            v3 Fs = V3(0, 0, 0), Ts = V3(0, 0, 0);
            for (int w = 0; w < BLK / 32; w++) {
                Fs.x += rF[0][w]; Fs.y += rF[1][w]; Fs.z += rF[2][w];
                Ts.x += rT[0][w]; Ts.y += rT[1][w]; Ts.z += rT[2][w];
            }
            Fs.z -= mp.mass * GRAV;
            v3 tb = V3(r0.x * Ts.x + r1.x * Ts.y + r2.x * Ts.z,
                       r0.y * Ts.x + r1.y * Ts.y + r2.y * Ts.z,
                       r0.z * Ts.x + r1.z * Ts.y + r2.z * Ts.z);
            v3 Iw = V3(mp.I[0] * wb.x + mp.I[3] * wb.y + mp.I[4] * wb.z,
                       mp.I[3] * wb.x + mp.I[1] * wb.y + mp.I[5] * wb.z,
                       mp.I[4] * wb.x + mp.I[5] * wb.y + mp.I[2] * wb.z);
            v3 gyro = cross(wb, Iw);
            v3 tq = sub(tb, gyro);
            v3 dw = V3(mp.invI[0] * tq.x + mp.invI[1] * tq.y + mp.invI[2] * tq.z,
                       mp.invI[3] * tq.x + mp.invI[4] * tq.y + mp.invI[5] * tq.z,
                       mp.invI[6] * tq.x + mp.invI[7] * tq.y + mp.invI[8] * tq.z);
            wb = add(wb, scl(dw, SIM_DT));
            float wl = len(wb); if (wl > 150.f) wb = scl(wb, 150.f / wl);
            vel = add(vel, scl(Fs, mp.invmass * SIM_DT));
            float vl = len(vel); if (vl > 60.f) vel = scl(vel, 60.f / vl);
            pos = add(pos, scl(vel, SIM_DT));
            quat dq = qmul(s_q, Q(0, wb.x, wb.y, wb.z));
            s_q = qnorm(Q(s_q.w + 0.5f * dq.w * SIM_DT, s_q.x + 0.5f * dq.x * SIM_DT,
                          s_q.y + 0.5f * dq.y * SIM_DT, s_q.z + 0.5f * dq.z * SIM_DT));
            s_vel = vel; s_wb = wb; s_pos = pos; s_t += SIM_DT;
            int bad = !(pos.x == pos.x && vel.x == vel.x);
            if (pos.z <= 0.0f || bad || fabsf(pos.y) > 300.f || pos.x < -50.f) {
                s_done = 1;
                fitness[p] = bad ? -10.0f : pos.x;
            } else if (step == SIM_MAX_STEPS - 1) {
                s_done = 1; fitness[p] = pos.x; // still airborne: credit distance so far
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        PlaneState s; s.pos = s_pos; s.q = s_q; s.vel = s_vel; s.wb = s_wb; s.done = s_done; s.fitness = 0;
        st[p] = s;
    }
}

void gpu_prepare(const Genome* d_gs, void* pbv, int pop) {
    PanelBuf* pb = (PanelBuf*)pbv;
    k_prepare<<<pop, 64>>>(d_gs, *pb, pop);
    CK(cudaGetLastError());
}

void* panelbuf_alloc(int pop) {
    PanelBuf* pb = new PanelBuf;
    CK(cudaMalloc(&pb->pc, sizeof(float4) * pop * MAX_T));
    CK(cudaMalloc(&pb->pn, sizeof(float4) * pop * MAX_T));
    CK(cudaMalloc(&pb->mp, sizeof(MassProps) * pop));
    CK(cudaMalloc(&pb->nt, sizeof(int) * pop));
    CK(cudaMalloc(&pb->st, sizeof(PlaneState) * pop));
    return pb;
}

// chunked launches (~1s sim each) keep kernels short: GSP/driver friendly, preemptible
void gpu_sim(void* pbv, int pop, int seed, float* d_fitness,
             float8* d_rec, int rec_every, int rec_cap, float spread) {
    PanelBuf* pb = (PanelBuf*)pbv;
    const int CHUNK = 600;
    for (int s0 = 0; s0 < SIM_MAX_STEPS; s0 += CHUNK) {
        k_sim<<<pop, BLK>>>(*pb, pop, seed, d_fitness, d_rec, rec_every, rec_cap, spread,
                            pb->st, s0, CHUNK);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
    }
}

// ---------------- GA (host) ----------------
static inline float gget(const Genome* g, int i) { return ((const float*)g)[i]; }
static inline void gset(Genome* g, int i, float v) { ((float*)g)[i] = fminf(G_HI[i], fmaxf(G_LO[i], v)); }

static void grand(Genome* g, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0, 1);
    for (int i = 0; i < GENOME_N; i++) gset(g, i, G_LO[i] + u(rng) * (G_HI[i] - G_LO[i]));
}

static void save_best(const char* out_dir, const Genome* g) {
    char path[512];
    snprintf(path, sizeof path, "%s/best.bin", out_dir);
    FILE* f = fopen(path, "wb"); fwrite(g, sizeof(Genome), 1, f); fclose(f);
}

static void save_genomes(const char* out_dir, const char* name, const Genome* gs, int n) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", out_dir, name);
    FILE* f = fopen(path, "wb");
    fwrite(&n, 4, 1, f); fwrite(gs, sizeof(Genome), n, f); fclose(f);
}

void run_ga(int pop, int gens, int nseeds, const char* out_dir, const char* init_path) {
    std::mt19937 rng(42);
    std::vector<Genome> gs(pop), next(pop);
    for (auto& g : gs) grand(&g, rng);
    if (init_path) {
        FILE* f = fopen(init_path, "rb");
        if (f) {
            int n = 0; fread(&n, 4, 1, f);
            if (n > pop) n = pop;
            fread(gs.data(), sizeof(Genome), n, f); fclose(f);
            printf("warm start: %d genomes from %s\n", n, init_path);
        }
    }
    Genome* d_gs; float* d_fit;
    CK(cudaMalloc(&d_gs, sizeof(Genome) * pop));
    CK(cudaMalloc(&d_fit, sizeof(float) * pop));
    void* pb = panelbuf_alloc(pop);
    std::vector<float> fit(pop), acc(pop);
    std::vector<int> idx(pop);
    Genome best_ever; float best_ever_f = -1e9f;
    std::vector<Genome> archive;
    char path[512];
    snprintf(path, sizeof path, "%s/history.csv", out_dir);
    FILE* hist = fopen(path, "w");
    fprintf(hist, "gen,best,mean,p90\n");
    std::uniform_real_distribution<float> u01(0, 1);
    std::normal_distribution<float> nrm(0, 1);
    for (int gen = 0; gen < gens; gen++) {
        CK(cudaMemcpy(d_gs, gs.data(), sizeof(Genome) * pop, cudaMemcpyHostToDevice));
        gpu_prepare(d_gs, pb, pop);
        std::fill(acc.begin(), acc.end(), 0.f);
        for (int s = 0; s < nseeds; s++) {
            gpu_sim(pb, pop, gen * nseeds + s + 1, d_fit, nullptr, 1, 0, 0);
            CK(cudaMemcpy(fit.data(), d_fit, sizeof(float) * pop, cudaMemcpyDeviceToHost));
            for (int i = 0; i < pop; i++) acc[i] += fit[i] / nseeds;
        }
        for (int i = 0; i < pop; i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return acc[a] > acc[b]; });
        float mean = 0; for (int i = 0; i < pop; i++) mean += acc[i] / pop;
        fprintf(hist, "%d,%.3f,%.3f,%.3f\n", gen, acc[idx[0]], mean, acc[idx[pop / 10]]);
        fflush(hist);
        if (gen % 10 == 0 || gen == gens - 1)
            printf("gen %4d best %7.2fm mean %7.2fm p90 %7.2fm\n", gen, acc[idx[0]], mean, acc[idx[pop / 10]]);
        if (acc[idx[0]] > best_ever_f) { best_ever_f = acc[idx[0]]; best_ever = gs[idx[0]]; }
        archive.push_back(gs[idx[0]]);
        if (gen % 25 == 24) { // checkpoint: GPU crash mid-run keeps artifacts
            save_best(out_dir, &best_ever);
            std::vector<Genome> srt(pop);
            for (int i = 0; i < pop; i++) srt[i] = gs[idx[i]];
            save_genomes(out_dir, "pop.bin", srt.data(), pop);
            save_genomes(out_dir, "archive.bin", archive.data(), (int)archive.size());
        }
        // evolve
        int elite = pop / 64;
        float anneal = 1.0f - 0.6f * gen / gens;
        for (int i = 0; i < elite; i++) next[i] = gs[idx[i]];
        for (int i = elite; i < pop; i++) {
            if (u01(rng) < 0.05f) { grand(&next[i], rng); continue; }
            auto tourn = [&]() {
                int b = idx[(int)(u01(rng) * pop)];
                for (int k = 0; k < 3; k++) { int c = (int)(u01(rng) * pop); if (acc[c] > acc[b]) b = c; }
                return b;
            };
            Genome pa = gs[tourn()], pc = gs[tourn()];
            for (int j = 0; j < GENOME_N; j++) {
                float a = -0.1f + 1.2f * u01(rng);
                float v = gget(&pa, j) * a + gget(&pc, j) * (1 - a);
                if (u01(rng) < 0.25f) v += nrm(rng) * 0.08f * (G_HI[j] - G_LO[j]) * anneal;
                gset(&next[i], j, v);
            }
        }
        gs.swap(next);
    }
    fclose(hist);
    save_best(out_dir, &best_ever);
    // final population sorted (for showcase) — re-eval last gen already in gs? use archive + last sorted set
    CK(cudaMemcpy(d_gs, gs.data(), sizeof(Genome) * pop, cudaMemcpyHostToDevice));
    gpu_prepare(d_gs, pb, pop);
    std::fill(acc.begin(), acc.end(), 0.f);
    for (int s = 0; s < nseeds; s++) {
        gpu_sim(pb, pop, 777 + s, d_fit, nullptr, 1, 0, 0);
        CK(cudaMemcpy(fit.data(), d_fit, sizeof(float) * pop, cudaMemcpyDeviceToHost));
        for (int i = 0; i < pop; i++) acc[i] += fit[i] / nseeds;
    }
    for (int i = 0; i < pop; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return acc[a] > acc[b]; });
    std::vector<Genome> srt(pop);
    for (int i = 0; i < pop; i++) srt[i] = gs[idx[i]];
    save_genomes(out_dir, "pop.bin", srt.data(), pop);
    save_genomes(out_dir, "archive.bin", archive.data(), (int)archive.size());
    printf("best ever: %.2fm (saved %s/best.bin)\n", best_ever_f, out_dir);
    Genome* bg = &best_ever;
    printf("genome: span=%.3f chord=%.3f taper=%.2f sweep=%.2f dih_in=%.2f dih_out=%.2f kink=%.2f\n"
           "        camber=(%.3f,%.3f) washout=%.3f elev=%.3f nose=%.2f keel=%.3f deck=(%.3f,%.2f) body=%.2f\n",
           bg->span, bg->chord, bg->taper, bg->sweep, bg->dihedral_in, bg->dihedral_out, bg->kink,
           bg->camber1, bg->camber2, bg->washout, bg->elevator, bg->nose_mass, bg->keel_h,
           bg->deck_gap, bg->deck_scale, bg->body_frac);
}
