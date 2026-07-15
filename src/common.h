#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------- vec/quat ----------------
struct v3 { float x, y, z; };
HD static inline v3 V3(float x, float y, float z) { v3 a; a.x = x; a.y = y; a.z = z; return a; }
HD static inline v3 add(v3 a, v3 b) { return V3(a.x + b.x, a.y + b.y, a.z + b.z); }
HD static inline v3 sub(v3 a, v3 b) { return V3(a.x - b.x, a.y - b.y, a.z - b.z); }
HD static inline v3 scl(v3 a, float s) { return V3(a.x * s, a.y * s, a.z * s); }
HD static inline float dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD static inline v3 cross(v3 a, v3 b) {
    return V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
HD static inline float len(v3 a) { return sqrtf(dot(a, a)); }
HD static inline v3 norm3(v3 a) { float l = len(a); return l > 1e-12f ? scl(a, 1.0f / l) : V3(0, 0, 1); }

struct quat { float w, x, y, z; };
HD static inline quat Q(float w, float x, float y, float z) { quat q; q.w = w; q.x = x; q.y = y; q.z = z; return q; }
HD static inline quat qmul(quat a, quat b) {
    return Q(a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
             a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
             a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
             a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}
HD static inline quat qnorm(quat q) {
    float l = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    float il = 1.0f / l; return Q(q.w * il, q.x * il, q.y * il, q.z * il);
}
// rotation matrix rows from quat (body->world)
HD static inline void qmat(quat q, v3* r0, v3* r1, v3* r2) {
    float w = q.w, x = q.x, y = q.y, z = q.z;
    *r0 = V3(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y));
    *r1 = V3(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x));
    *r2 = V3(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y));
}
HD static inline v3 qrot(quat q, v3 v) {
    v3 r0, r1, r2; qmat(q, &r0, &r1, &r2);
    return V3(dot(r0, v), dot(r1, v), dot(r2, v));
}
HD static inline v3 qrot_inv(quat q, v3 v) {
    v3 r0, r1, r2; qmat(q, &r0, &r1, &r2); // transpose apply
    return V3(r0.x * v.x + r1.x * v.y + r2.x * v.z,
              r0.y * v.x + r1.y * v.y + r2.y * v.z,
              r0.z * v.x + r1.z * v.y + r2.z * v.z);
}
HD static inline quat qaxis(v3 axis, float ang) {
    float s = sinf(ang * 0.5f); v3 a = norm3(axis);
    return Q(cosf(ang * 0.5f), a.x * s, a.y * s, a.z * s);
}

// ---------------- genome ----------------
#define GENOME_N 16
struct Genome {
    float span;        // wingspan along surface [m]
    float chord;       // root chord [m]
    float taper;       // tip/root chord ratio
    float sweep;       // leading-edge sweep [rad]
    float dihedral_in; // inner panel dihedral [rad]
    float dihedral_out;// outer panel dihedral (winglets) [rad]
    float kink;        // spanwise fraction where outer panel begins
    float camber1;     // bezier ctrl 1 (fwd camber, frac of chord)
    float camber2;     // bezier ctrl 2 (aft camber / reflex)
    float washout;     // tip twist [rad]
    float elevator;    // trailing-edge reflex slope
    float nose_mass;   // ballast at nose, x paper mass
    float keel_h;      // keel depth [m]
    float deck_gap;    // biplane deck gap [m], <0.02 => monoplane
    float deck_scale;  // upper deck scale
    float body_frac;   // flat center body width fraction
};
static const float G_LO[GENOME_N] = { 0.12f, 0.06f, 0.25f, 0.00f, -0.30f, -1.20f, 0.45f, -0.04f, -0.06f, -0.20f, -0.20f, 0.0f, 0.005f, 0.000f, 0.50f, 0.06f };
static const float G_HI[GENOME_N] = { 0.42f, 0.24f, 1.00f, 0.90f,  0.60f,  1.20f, 0.90f,  0.10f,  0.06f,  0.08f,  0.20f, 1.5f, 0.050f, 0.060f, 1.00f, 0.30f };

// ---------------- mesh ----------------
#define CH 5            // chordwise segments
#define SP 8            // spanwise stations per half
#define MAX_V 512
#define MAX_T 900
#define PAPER_RHO 0.08f // kg/m^2 (80 gsm)

struct MassProps {
    float mass, invmass;
    v3 com;
    float I[6];    // xx yy zz xy xz yz (about com, body frame)
    float invI[9]; // full 3x3 inverse
};

HD static inline float camber_z(const Genome* g, float xc) {
    // cubic bezier chord-line: P0(0,0) P1(0.33,c1) P2(0.75,c2) P3(1, te)
    float te = g->elevator * 0.22f;
    float u = xc, v = 1.0f - u;
    return 3 * v * v * u * g->camber1 + 3 * v * u * u * g->camber2 + u * u * u * te;
}

// spanwise dihedral polyline: arc-length t -> (y,z)
HD static inline void span_yz(const Genome* g, float t, float* y, float* z) {
    float half = g->span * 0.5f;
    float b = g->body_frac * half, k = g->kink * half;
    if (k < b) k = b;
    float yy = 0, zz = 0, seg;
    seg = fminf(t, b);              yy += seg;                          // body flat
    seg = fminf(t, k) - b; if (seg > 0) { yy += seg * cosf(g->dihedral_in);  zz += seg * sinf(g->dihedral_in); }
    seg = t - k;           if (seg > 0) { yy += seg * cosf(g->dihedral_out); zz += seg * sinf(g->dihedral_out); }
    *y = yy; *z = zz;
}

// wing surface point: u in [-1,1] spanwise, xc in [0,1] chordwise (LE->TE)
HD static inline v3 wing_pt(const Genome* g, float u, float xc, float zoff, float scale) {
    float half = g->span * 0.5f * scale;
    float t = fabsf(u) * half;
    float y, z; span_yz(g, t, &y, &z);
    float tf = fabsf(u);
    float c = g->chord * scale * (1.0f + (g->taper - 1.0f) * tf);
    float x_le = 0.5f * g->chord * scale - tanf(g->sweep) * t;
    float x = x_le - xc * c;
    float zc = z * scale + camber_z(g, xc) * c;
    float x_c4 = x_le - 0.25f * c;
    float tw = g->washout * tf;
    zc += tw * (x - x_c4); // small-angle twist about quarter chord
    return V3(x, (u < 0 ? -y : y), zc + zoff);
}

struct MeshOut { int nv, nt; };

// emit grid of (2*SP+1) x (CH+1) verts as triangles
HD static inline void emit_wing(const Genome* g, v3* V, int (*T)[3], int* nv, int* nt, float zoff, float scale) {
    int base = *nv;
    for (int i = 0; i <= 2 * SP; i++) {
        float u = (float)(i - SP) / SP;
        for (int j = 0; j <= CH; j++) {
            float xc = (float)j / CH;
            V[(*nv)++] = wing_pt(g, u, xc, zoff, scale);
        }
    }
    for (int i = 0; i < 2 * SP; i++)
        for (int j = 0; j < CH; j++) {
            int a = base + i * (CH + 1) + j, b = a + CH + 1;
            T[*nt][0] = a; T[*nt][1] = b;     T[*nt][2] = a + 1; (*nt)++;
            T[*nt][0] = b; T[*nt][1] = b + 1; T[*nt][2] = a + 1; (*nt)++;
        }
}

HD static inline void build_mesh(const Genome* g, v3* V, int (*T)[3], MeshOut* mo) {
    int nv = 0, nt = 0;
    emit_wing(g, V, T, &nv, &nt, 0, 1.0f);
    int decks = g->deck_gap > 0.02f ? 2 : 1;
    if (decks == 2) emit_wing(g, V, T, &nv, &nt, g->deck_gap, g->deck_scale);
    // keel: vertical sheet under centerline
    {
        int base = nv;
        for (int j = 0; j <= 3; j++) {
            float xc = (float)j / 3;
            v3 top = wing_pt(g, 0, xc, 0, 1.0f);
            V[nv++] = top;
            V[nv++] = V3(top.x, 0, top.z - g->keel_h * (0.35f + 0.65f * xc));
        }
        for (int j = 0; j < 3; j++) {
            int a = base + j * 2;
            T[nt][0] = a; T[nt][1] = a + 1; T[nt][2] = a + 2; nt++;
            T[nt][0] = a + 1; T[nt][1] = a + 3; T[nt][2] = a + 2; nt++;
        }
    }
    // biplane struts at kink stations
    if (decks == 2) {
        for (int s = -1; s <= 1; s += 2) {
            float u = s * g->kink;
            int base = nv;
            v3 lo0 = wing_pt(g, u, 0.25f, 0, 1.0f), lo1 = wing_pt(g, u, 0.55f, 0, 1.0f);
            v3 hi0 = wing_pt(g, u * 1.0f, 0.25f, g->deck_gap, g->deck_scale);
            v3 hi1 = wing_pt(g, u * 1.0f, 0.55f, g->deck_gap, g->deck_scale);
            V[nv++] = lo0; V[nv++] = lo1; V[nv++] = hi0; V[nv++] = hi1;
            T[nt][0] = base; T[nt][1] = base + 1; T[nt][2] = base + 2; nt++;
            T[nt][0] = base + 1; T[nt][1] = base + 3; T[nt][2] = base + 2; nt++;
        }
    }
    mo->nv = nv; mo->nt = nt;
}

HD static inline v3 nose_point(const Genome* g) { return V3(0.5f * g->chord, 0, 0); }

HD static inline void mass_props(const Genome* g, const v3* V, const int (*T)[3], int nt, MassProps* mp) {
    float m = 0; v3 c = V3(0, 0, 0);
    for (int i = 0; i < nt; i++) {
        v3 a = V[T[i][0]], b = V[T[i][1]], d = V[T[i][2]];
        float A = 0.5f * len(cross(sub(b, a), sub(d, a)));
        float mi = A * PAPER_RHO;
        v3 ct = scl(add(add(a, b), d), 1.0f / 3);
        m += mi; c = add(c, scl(ct, mi));
    }
    float paper_m = m;
    v3 np = nose_point(g);
    float mn = g->nose_mass * paper_m;
    m += mn; c = add(c, scl(np, mn));
    c = scl(c, 1.0f / m);
    float I[6] = { 0, 0, 0, 0, 0, 0 };
#define ACC_I(r, mi) { I[0]+=mi*((r).y*(r).y+(r).z*(r).z); I[1]+=mi*((r).x*(r).x+(r).z*(r).z); \
    I[2]+=mi*((r).x*(r).x+(r).y*(r).y); I[3]-=mi*(r).x*(r).y; I[4]-=mi*(r).x*(r).z; I[5]-=mi*(r).y*(r).z; }
    for (int i = 0; i < nt; i++) {
        v3 a = V[T[i][0]], b = V[T[i][1]], d = V[T[i][2]];
        float A = 0.5f * len(cross(sub(b, a), sub(d, a)));
        float mi = A * PAPER_RHO;
        v3 r = sub(scl(add(add(a, b), d), 1.0f / 3), c);
        ACC_I(r, mi);
    }
    { v3 r = sub(np, c); ACC_I(r, mn); }
#undef ACC_I
    // regularize + invert symmetric 3x3
    float reg = 1e-9f;
    float xx = I[0] + reg, yy = I[1] + reg, zz = I[2] + reg, xy = I[3], xz = I[4], yz = I[5];
    float det = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz);
    float id = 1.0f / det;
    mp->invI[0] = (yy * zz - yz * yz) * id; mp->invI[1] = (xz * yz - xy * zz) * id; mp->invI[2] = (xy * yz - xz * yy) * id;
    mp->invI[3] = mp->invI[1]; mp->invI[4] = (xx * zz - xz * xz) * id; mp->invI[5] = (xy * xz - xx * yz) * id;
    mp->invI[6] = mp->invI[2]; mp->invI[7] = mp->invI[5]; mp->invI[8] = (xx * yy - xy * xy) * id;
    mp->I[0] = xx; mp->I[1] = yy; mp->I[2] = zz; mp->I[3] = xy; mp->I[4] = xz; mp->I[5] = yz;
    mp->mass = m; mp->invmass = 1.0f / m; mp->com = c;
}

// ---------------- wind ----------------
HD static inline v3 wind_at(v3 p, float t, int seed) {
    float s = (float)(seed * 37 % 97);
    float gx = 0.55f * sinf(0.9f * p.x + 1.3f * t + s) + 0.35f * sinf(2.3f * p.z + 2.1f * t + 1.7f * s)
             + 0.25f * sinf(4.1f * p.y + 3.3f * t + 0.4f * s);
    float gy = 0.50f * sinf(1.1f * p.x + 1.7f * t + 2.2f * s) + 0.30f * sinf(3.1f * p.z + 2.9f * t + 0.9f * s);
    float gz = 0.30f * sinf(1.4f * p.x + 2.3f * t + 3.1f * s) + 0.20f * sinf(2.7f * p.y + 1.9f * t + 1.2f * s);
    float base_x = -0.35f + 0.25f * sinf(s * 0.71f); // light headwind component
    float base_y = 0.30f * sinf(s * 1.13f);
    return V3(base_x + gx, base_y + gy, gz);
}

// ---------------- sim params ----------------
#define SIM_DT (1.0f / 600.0f)
#define SIM_MAX_STEPS 7200      // 12 s
#define LAUNCH_H 1.8f
#define LAUNCH_V 9.5f
#define LAUNCH_PITCH 0.10f      // rad
#define AIR_RHO 1.225f
#define CN_SLOPE 1.9f           // attached-flow normal-force slope per panel
#define CF_FRICTION 0.02f
#define GRAV 9.81f

struct PlaneState { v3 pos; quat q; v3 vel; v3 wb; float done; float fitness; };
