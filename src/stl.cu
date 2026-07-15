#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>

// solid shell: offset sheet +-th/2 along area-weighted vertex normals, stitch boundary edges
void export_stl(const Genome* g, const char* path, float thickness_mm) {
    v3 V[MAX_V]; int T[MAX_T][3]; MeshOut mo;
    build_mesh(g, V, T, &mo);
    MassProps mp; mass_props(g, V, T, mo.nt, &mp);
    v3 ext = mesh_extent(V, mo.nv);
    if (!fits_print_volume(V, mo.nv, PRINT_VOLUME_M - thickness_mm * 0.001f)) {
        fprintf(stderr, "refusing STL outside 250 mm build volume: %.1f x %.1f x %.1f mm\n",
                ext.x * 1000, ext.y * 1000, ext.z * 1000);
        return;
    }
    for (int i = 0; i < mo.nv; i++) V[i] = sub(V[i], mp.com);
    std::vector<v3> vn(mo.nv, V3(0, 0, 0));
    for (int t = 0; t < mo.nt; t++) {
        v3 n = cross(sub(V[T[t][1]], V[T[t][0]]), sub(V[T[t][2]], V[T[t][0]]));
        for (int k = 0; k < 3; k++) {
            // sign-align contributions so folded sheets keep consistent offset
            v3 cur = vn[T[t][k]];
            if (dot(cur, n) < 0 && dot(cur, cur) > 1e-16f) vn[T[t][k]] = sub(cur, n);
            else vn[T[t][k]] = add(cur, n);
        }
    }
    float h = thickness_mm * 0.0005f; // half thickness in meters
    std::vector<v3> up(mo.nv), dn(mo.nv);
    for (int i = 0; i < mo.nv; i++) {
        v3 n = norm3(vn[i]);
        up[i] = add(V[i], scl(n, h));
        dn[i] = sub(V[i], scl(n, h));
    }
    struct Tri { v3 a, b, c; };
    std::vector<Tri> out;
    for (int t = 0; t < mo.nt; t++) {
        out.push_back({ up[T[t][0]], up[T[t][1]], up[T[t][2]] });
        out.push_back({ dn[T[t][0]], dn[T[t][2]], dn[T[t][1]] });
    }
    // boundary edges (used once)
    std::map<std::pair<int, int>, int> ec;
    for (int t = 0; t < mo.nt; t++)
        for (int k = 0; k < 3; k++) {
            int a = T[t][k], b = T[t][(k + 1) % 3];
            ec[{ a < b ? a : b, a < b ? b : a }]++;
        }
    for (int t = 0; t < mo.nt; t++)
        for (int k = 0; k < 3; k++) {
            int a = T[t][k], b = T[t][(k + 1) % 3];
            if (ec[{ a < b ? a : b, a < b ? b : a }] == 1) {
                out.push_back({ up[a], dn[a], up[b] });
                out.push_back({ dn[a], dn[b], up[b] });
            }
        }
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    char hdr[80] = "planes-rl evolved printable PLA glider (units: mm)";
    fwrite(hdr, 80, 1, f);
    uint32_t n = (uint32_t)out.size();
    fwrite(&n, 4, 1, f);
    for (auto& tr : out) {
        v3 nn = norm3(cross(sub(tr.b, tr.a), sub(tr.c, tr.a)));
        float rec[12] = { nn.x, nn.y, nn.z,
                          tr.a.x * 1000, tr.a.y * 1000, tr.a.z * 1000,
                          tr.b.x * 1000, tr.b.y * 1000, tr.b.z * 1000,
                          tr.c.x * 1000, tr.c.y * 1000, tr.c.z * 1000 };
        fwrite(rec, 48, 1, f);
        uint16_t attr = 0; fwrite(&attr, 2, 1, f);
    }
    fclose(f);
    float actual_mass = mp.mass * (thickness_mm * 0.001f / PLA_WALL_M);
    printf("stl: %u tris -> %s (PLA %.1f g @ %.2f mm, bounds %.0f x %.0f x %.0f mm)\n",
           n, path, actual_mass * 1000, thickness_mm,
           ext.x * 1000, ext.y * 1000, ext.z * 1000);
}
