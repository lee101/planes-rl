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
#define GENOME_N 35
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
    float nose_mass;   // ballast at nose, x PLA shell mass
    float keel_h;      // keel depth [m]
    float deck_gap;    // biplane deck gap [m], <0.02 => monoplane
    float deck_scale;  // upper deck scale
    float body_frac;   // flat center body width fraction
    float fin_height;  // vertical stabilizer height [m]
    float fin_chord;   // fin root chord / wing root chord
    float fin_sweep;   // fin leading-edge rearward sweep / fin root chord
    float tail_span;   // horizontal tail span / wing span
    float tail_chord;  // horizontal tail chord / wing root chord
    float tail_angle;  // horizontal tail incidence [rad]
    float wing_le_b1;  // cubic Bezier leading-edge control 1 / root chord
    float wing_le_b2;  // cubic Bezier leading-edge control 2 / root chord
    float chord_b1;    // cubic chord-distribution control 1
    float chord_b2;    // cubic chord-distribution control 2
    float fin_layout;  // center, twin, triple, or outboard symmetric fins
    float fin_cant;    // paired-fin outward cant [rad]
    float fin_taper;   // fin tip/root chord ratio
    float fin_span_pos;// paired-fin position / tail semispan
    float tail_sweep;  // horizontal-tail rearward sweep / tail chord
    float tail_taper;  // horizontal-tail tip/root chord ratio
    float tail_dihedral;// horizontal-tail dihedral [rad]
    float wing_tip_round; // elliptical-tip blend
    float center_chord;// root/center chord expansion
};
static const float G_LO[GENOME_N] = { 0.12f,0.05f,0.04f,0.00f,-0.30f,-1.20f,0.45f,-0.04f,-0.06f,-0.20f,-0.20f,0.0f,0.005f,0.000f,0.50f,0.06f,
                                      0.004f,0.12f,0.00f,0.12f,0.10f,-0.16f,
                                     -0.80f,-0.80f,-0.65f,-0.65f,0.00f,0.00f,0.10f,0.15f,-0.40f,0.15f,-0.50f,0.00f,-0.25f };
static const float G_HI[GENOME_N] = { 0.25f,0.20f,1.00f,0.90f, 0.60f, 1.20f,0.90f, 0.10f, 0.06f, 0.08f, 0.20f,1.5f,0.050f,0.060f,1.00f,0.30f,
                                      0.065f,0.65f,0.85f,0.65f,0.45f, 0.16f,
                                      0.80f, 0.80f, 1.00f, 1.00f,1.00f,0.75f,1.00f,1.00f, 1.00f,1.00f, 0.50f,1.00f, 0.80f };

// ---------------- mesh ----------------
#define CH 5            // chordwise segments
#define SP 8            // spanwise stations per half
#define MAX_V 512
#define MAX_T 900
// Printable shell model.  A single 0.45 mm PLA wall is the default optimization
// target; STL export may use another thickness and reports the corresponding mass.
#define PLA_DENSITY 1240.0f       // kg/m^3, ordinary rigid PLA
#define PLA_WALL_M 0.00045f       // 0.45 mm extrusion-width shell
#define SHELL_AREAL_RHO (PLA_DENSITY * PLA_WALL_M)
#define PRINT_VOLUME_M 0.250f     // 250 x 250 x 250 mm build envelope

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
    // Cubic Bezier planform. Mirroring |u| is a hard symmetry heuristic: the
    // optimizer can invent strongly curved, elliptical, plank, swept, or delta
    // wings without wasting samples on accidental left/right mismatches.
    float v = 1.0f - tf;
    float cr = 1.0f + g->center_chord;
    float ct = g->taper;
    float c1 = 1.0f + g->chord_b1, c2 = ct + g->chord_b2;
    float ratio = v*v*v*cr + 3*v*v*tf*c1 + 3*v*tf*tf*c2 + tf*tf*tf*ct;
    float ellipse = sqrtf(fmaxf(0.08f, 1.0f - 0.92f*tf*tf));
    ratio *= 1.0f - g->wing_tip_round + g->wing_tip_round * ellipse;
    ratio = fmaxf(0.035f, fminf(1.9f, ratio));
    float c = g->chord * scale * ratio;
    float xr = 0.5f * g->chord * scale * cr;
    float xt = 0.5f * g->chord * scale - tanf(g->sweep) * half;
    float xb1 = xr + g->wing_le_b1 * g->chord * scale;
    float xb2 = xt + g->wing_le_b2 * g->chord * scale;
    float x_le = v*v*v*xr + 3*v*v*tf*xb1 + 3*v*tf*tf*xb2 + tf*tf*tf*xt;
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

// Small, evolvable empennage attached at the root trailing edge.  It gives the
// search independent yaw and pitch stability controls instead of forcing the
// swept wing and ventral keel to do every job.
HD static inline void emit_one_fin(const Genome* g, v3* V, int (*T)[3], int* nv, int* nt,
                                   float y0, float z0, int side) {
    float fc = g->chord * g->fin_chord;
    float sweep = fc * g->fin_sweep;
    float xr = -0.32f * g->chord - g->tail_chord * g->chord * 0.82f;
    float tipc = fc * g->fin_taper;
    float cant = side == 0 ? 0.0f : g->fin_cant;
    float ya = y0 + side * sinf(cant) * g->fin_height;
    float za = z0 + cosf(cant) * g->fin_height;
    int base = *nv;
    V[(*nv)++] = V3(xr + 0.5f*fc, y0, z0);
    V[(*nv)++] = V3(xr - 0.5f*fc, y0, z0);
    V[(*nv)++] = V3(xr - sweep + 0.5f*tipc, ya, za);
    V[(*nv)++] = V3(xr - sweep - 0.5f*tipc, ya, za);
    T[*nt][0]=base; T[*nt][1]=base+1; T[*nt][2]=base+2; (*nt)++;
    T[*nt][0]=base+1; T[*nt][1]=base+3; T[*nt][2]=base+2; (*nt)++;
}

HD static inline void emit_tail(const Genome* g, v3* V, int (*T)[3], int* nv, int* nt) {
    float hs = 0.5f * g->span * g->tail_span;
    float hc = g->chord * g->tail_chord;
    float x0 = -0.32f * g->chord;
    float z0 = camber_z(g, 0.82f) * g->chord;
    int base = *nv;
    for (int i = 0; i < 3; i++) {
        float u = (float)(i - 1), au = fabsf(u);
        float tc = hc * (1.0f + (g->tail_taper - 1.0f) * au);
        float xle = x0 - g->tail_sweep * hc * au;
        for (int j = 0; j < 3; j++) {
            float xc = 0.5f * j;
            float x = xle - xc * tc;
            float z = z0 + g->tail_angle * (x - xle) + au * hs * sinf(g->tail_dihedral);
            V[(*nv)++] = V3(x, u * hs * cosf(g->tail_dihedral), z);
        }
    }
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
        int a = base + i * 3 + j, b = a + 3;
        T[*nt][0] = a; T[*nt][1] = b; T[*nt][2] = a + 1; (*nt)++;
        T[*nt][0] = b; T[*nt][1] = b + 1; T[*nt][2] = a + 1; (*nt)++;
    }

    float yp = hs * g->fin_span_pos * cosf(g->tail_dihedral);
    float zp = z0 + hs * g->fin_span_pos * sinf(g->tail_dihedral);
    if (g->fin_layout < 0.25f) {
        emit_one_fin(g,V,T,nv,nt,0,z0,0);                 // classic center fin
    } else if (g->fin_layout < 0.50f) {
        emit_one_fin(g,V,T,nv,nt,-yp,zp,-1);             // symmetric twin fins
        emit_one_fin(g,V,T,nv,nt, yp,zp, 1);
    } else if (g->fin_layout < 0.75f) {
        emit_one_fin(g,V,T,nv,nt,0,z0,0);                // symmetric triple fin
        emit_one_fin(g,V,T,nv,nt,-yp,zp,-1);
        emit_one_fin(g,V,T,nv,nt, yp,zp, 1);
    } else {
        float yo = hs * 0.94f * cosf(g->tail_dihedral);  // outboard endplate fins
        float zo = z0 + hs * 0.94f * sinf(g->tail_dihedral);
        emit_one_fin(g,V,T,nv,nt,-yo,zo,-1);
        emit_one_fin(g,V,T,nv,nt, yo,zo, 1);
    }
}

HD static inline void build_mesh(const Genome* g, v3* V, int (*T)[3], MeshOut* mo) {
    int nv = 0, nt = 0;
    emit_wing(g, V, T, &nv, &nt, 0, 1.0f);
    emit_tail(g, V, T, &nv, &nt);
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

HD static inline v3 mesh_extent(const v3* V, int nv) {
    if (nv <= 0) return V3(0, 0, 0);
    v3 lo = V[0], hi = V[0];
    for (int i = 1; i < nv; i++) {
        lo.x = fminf(lo.x, V[i].x); lo.y = fminf(lo.y, V[i].y); lo.z = fminf(lo.z, V[i].z);
        hi.x = fmaxf(hi.x, V[i].x); hi.y = fmaxf(hi.y, V[i].y); hi.z = fmaxf(hi.z, V[i].z);
    }
    return sub(hi, lo);
}

HD static inline int fits_print_volume(const v3* V, int nv, float limit_m) {
    v3 e = mesh_extent(V, nv);
    return e.x <= limit_m && e.y <= limit_m && e.z <= limit_m;
}

HD static inline v3 nose_point(const Genome* g) { return V3(0.5f * g->chord, 0, 0); }

HD static inline void mass_props(const Genome* g, const v3* V, const int (*T)[3], int nt, MassProps* mp) {
    float m = 0; v3 c = V3(0, 0, 0);
    for (int i = 0; i < nt; i++) {
        v3 a = V[T[i][0]], b = V[T[i][1]], d = V[T[i][2]];
        float A = 0.5f * len(cross(sub(b, a), sub(d, a)));
        float mi = A * SHELL_AREAL_RHO;
        v3 ct = scl(add(add(a, b), d), 1.0f / 3);
        m += mi; c = add(c, scl(ct, mi));
    }
    float shell_m = m;
    v3 np = nose_point(g);
    float mn = g->nose_mass * shell_m;
    m += mn; c = add(c, scl(np, mn));
    c = scl(c, 1.0f / m);
    float I[6] = { 0, 0, 0, 0, 0, 0 };
#define ACC_I(r, mi) { I[0]+=mi*((r).y*(r).y+(r).z*(r).z); I[1]+=mi*((r).x*(r).x+(r).z*(r).z); \
    I[2]+=mi*((r).x*(r).x+(r).y*(r).y); I[3]-=mi*(r).x*(r).y; I[4]-=mi*(r).x*(r).z; I[5]-=mi*(r).y*(r).z; }
    for (int i = 0; i < nt; i++) {
        v3 a = V[T[i][0]], b = V[T[i][1]], d = V[T[i][2]];
        float A = 0.5f * len(cross(sub(b, a), sub(d, a)));
        float mi = A * SHELL_AREAL_RHO;
        v3 r = sub(scl(add(add(a, b), d), 1.0f / 3), c);
        // Exact second moment of a uniform triangular lamina, rather than
        // concentrating every panel at its centroid.
        v3 ra = sub(a, c), rb = sub(b, c), rd = sub(d, c);
        float exx = (ra.x*ra.x + rb.x*rb.x + rd.x*rd.x + ra.x*rb.x + ra.x*rd.x + rb.x*rd.x) / 6.0f;
        float eyy = (ra.y*ra.y + rb.y*rb.y + rd.y*rd.y + ra.y*rb.y + ra.y*rd.y + rb.y*rd.y) / 6.0f;
        float ezz = (ra.z*ra.z + rb.z*rb.z + rd.z*rd.z + ra.z*rb.z + ra.z*rd.z + rb.z*rd.z) / 6.0f;
        float exy = (2*(ra.x*ra.y + rb.x*rb.y + rd.x*rd.y) +
                     ra.x*rb.y + rb.x*ra.y + ra.x*rd.y + rd.x*ra.y + rb.x*rd.y + rd.x*rb.y) / 12.0f;
        float exz = (2*(ra.x*ra.z + rb.x*rb.z + rd.x*rd.z) +
                     ra.x*rb.z + rb.x*ra.z + ra.x*rd.z + rd.x*ra.z + rb.x*rd.z + rd.x*rb.z) / 12.0f;
        float eyz = (2*(ra.y*ra.z + rb.y*rb.z + rd.y*rd.z) +
                     ra.y*rb.z + rb.y*ra.z + ra.y*rd.z + rd.y*ra.z + rb.y*rd.z + rd.y*rb.z) / 12.0f;
        I[0] += mi * (eyy + ezz); I[1] += mi * (exx + ezz); I[2] += mi * (exx + eyy);
        I[3] -= mi * exy; I[4] -= mi * exz; I[5] -= mi * eyz;
        // Through-thickness inertia of the extruded PLA sheet.
        v3 nn = norm3(cross(sub(b, a), sub(d, a)));
        float th = mi * PLA_WALL_M * PLA_WALL_M / 12.0f;
        I[0] += th * (1.0f - nn.x*nn.x); I[1] += th * (1.0f - nn.y*nn.y); I[2] += th * (1.0f - nn.z*nn.z);
        I[3] -= th * nn.x*nn.y; I[4] -= th * nn.x*nn.z; I[5] -= th * nn.y*nn.z;
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

// ---------------- randomized environment ----------------
HD static inline uint32_t mix_bits(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
HD static inline float scenario_u01(int seed, int lane) {
    return (mix_bits((uint32_t)seed * 0x9e3779b9u + (uint32_t)lane * 0x85ebca6bu) >> 8) * (1.0f / 16777216.0f);
}
HD static inline float scenario_signed(int seed, int lane) { return scenario_u01(seed, lane) * 2.0f - 1.0f; }

struct ScenarioDesc {
    float speed, pitch, yaw, roll, height;
    float roll_rate, pitch_rate, yaw_rate;
    float air_rho, skin_cf, gust, base_x, base_y;
    float force_pitch, force_yaw; // impulse direction relative to held attitude
    float grip_x, grip_y, grip_z; // hand-force point relative to nominal CG [m]
    float material_scale;         // density x extrusion/flow variation
    float infill_ratio;           // added internal/rib mass / shell mass
    float print_bias_x, print_bias_y, print_bias_z; // asymmetric infill/adhesion
    float warp_pitch, warp_roll;  // small print-induced surface normal errors
};

HD static inline ScenarioDesc scenario_desc(int seed) {
    ScenarioDesc d;
    d.speed = 9.2f + 2.0f * scenario_signed(seed, 0);
    d.pitch = 0.10f + 0.20f * scenario_signed(seed, 1);
    d.yaw = 0.12f * scenario_signed(seed, 2);
    d.roll = 0.16f * scenario_signed(seed, 3);
    d.height = 1.75f + 0.35f * scenario_signed(seed, 4);
    d.roll_rate = 1.8f * scenario_signed(seed, 5);
    d.pitch_rate = 1.2f * scenario_signed(seed, 6);
    d.yaw_rate = 1.5f * scenario_signed(seed, 7);
    d.air_rho = 1.225f * (0.88f + 0.22f * scenario_u01(seed, 8));
    d.skin_cf = 0.02f * (0.75f + 0.55f * scenario_u01(seed, 9));
    d.gust = 0.45f + 1.10f * (0.5f + 0.5f * sinf(seed * 1.61803f));
    float phase = (float)(seed * 37 % 97);
    d.base_x = -0.25f + 1.15f * sinf(phase * 0.71f);
    d.base_y = 1.00f * sinf(phase * 1.13f);
    d.force_pitch = 0.075f * scenario_signed(seed, 10);
    d.force_yaw = 0.060f * scenario_signed(seed, 11);
    d.grip_x = 0.018f * scenario_signed(seed, 12);
    d.grip_y = 0.022f * scenario_signed(seed, 13);
    d.grip_z = 0.010f * scenario_signed(seed, 14);
    // PLA brand, moisture, flow calibration, wall width and sparse infill/ribs.
    d.material_scale = 0.88f + 0.24f * scenario_u01(seed, 15);
    d.infill_ratio = 0.06f + 0.34f * scenario_u01(seed, 16);
    d.print_bias_x = 0.018f * scenario_signed(seed, 17);
    d.print_bias_y = 0.010f * scenario_signed(seed, 18);
    d.print_bias_z = 0.006f * scenario_signed(seed, 19);
    // Higher/lower layer ridging is represented jointly by drag and mild warp.
    d.skin_cf *= 0.90f + 0.25f * scenario_u01(seed, 20);
    d.warp_pitch = 0.025f * scenario_signed(seed, 21);
    d.warp_roll = 0.035f * scenario_signed(seed, 22);
    return d;
}

HD static inline void point_inertia(float I[6], v3 r, float m) {
    I[0] += m * (r.y*r.y + r.z*r.z); I[1] += m * (r.x*r.x + r.z*r.z);
    I[2] += m * (r.x*r.x + r.y*r.y); I[3] -= m * r.x*r.y;
    I[4] -= m * r.x*r.z; I[5] -= m * r.y*r.z;
}

HD static inline void invert_inertia(MassProps* mp) {
    float xx=mp->I[0], yy=mp->I[1], zz=mp->I[2], xy=mp->I[3], xz=mp->I[4], yz=mp->I[5];
    float det = xx*(yy*zz-yz*yz) - xy*(xy*zz-yz*xz) + xz*(xy*yz-yy*xz);
    float id = 1.0f / fmaxf(det, 1e-24f);
    mp->invI[0]=(yy*zz-yz*yz)*id; mp->invI[1]=(xz*yz-xy*zz)*id; mp->invI[2]=(xy*yz-xz*yy)*id;
    mp->invI[3]=mp->invI[1]; mp->invI[4]=(xx*zz-xz*xz)*id; mp->invI[5]=(xy*xz-xx*yz)*id;
    mp->invI[6]=mp->invI[2]; mp->invI[7]=mp->invI[5]; mp->invI[8]=(xx*yy-xy*xy)*id;
}

// Convert slicer/process uncertainty into effective mass, CG and inertia.  The
// infill term follows the shell distribution but may be biased by imperfect
// adhesion, seams and nonuniform internal ribs.
HD static inline MassProps printed_mass_props(MassProps src, ScenarioDesc d, v3* cg_shift) {
    float m_shell = src.mass * d.material_scale;
    float m_fill = m_shell * d.infill_ratio;
    float m = m_shell + m_fill;
    v3 bias = V3(d.print_bias_x, d.print_bias_y, d.print_bias_z);
    *cg_shift = scl(bias, m_fill / m);
    float scale = m / src.mass;
    for (int k = 0; k < 6; k++) src.I[k] *= scale;
    // Parallel-axis correction for two distributed components whose centroids
    // are separated by the randomized print bias.
    point_inertia(src.I, scl(*cg_shift, -1.0f), m_shell);
    point_inertia(src.I, sub(bias, *cg_shift), m_fill);
    src.mass = m; src.invmass = 1.0f / m;
    src.com = add(src.com, *cg_shift);
    invert_inertia(&src);
    return src;
}

HD static inline v3 wind_at(v3 p, float t, int seed) {
    float s = (float)(seed * 37 % 97);
    ScenarioDesc d = scenario_desc(seed);
    float gx = d.gust * (0.55f * sinf(0.9f * p.x + 1.3f * t + s) + 0.35f * sinf(2.3f * p.z + 2.1f * t + 1.7f * s)
             + 0.25f * sinf(4.1f * p.y + 3.3f * t + 0.4f * s));
    float gy = d.gust * (0.50f * sinf(1.1f * p.x + 1.7f * t + 2.2f * s) + 0.30f * sinf(3.1f * p.z + 2.9f * t + 0.9f * s));
    float gz = d.gust * (0.30f * sinf(1.4f * p.x + 2.3f * t + 3.1f * s) + 0.20f * sinf(2.7f * p.y + 1.9f * t + 1.2f * s));
    return V3(d.base_x + gx, d.base_y + gy, gz);
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
