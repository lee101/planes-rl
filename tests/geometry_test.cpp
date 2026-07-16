#include "../src/common.h"
#include <assert.h>
#include <filesystem>
#include <stdio.h>
#include <vector>

void write_presets(const char*, float);

int main() {
    const char* dir = "out/host-geometry-test";
    std::filesystem::remove_all(dir);
    write_presets(dir, 0.45f);
    std::filesystem::path seeds = std::filesystem::path(dir) / "seeds.bin";
    FILE* f = fopen(seeds.string().c_str(), "rb");
    assert(f);
    int n = 0; assert(fread(&n, sizeof n, 1, f) == 1); assert(n == 5);
    std::vector<Genome> gs(n); assert(fread(gs.data(), sizeof(Genome), n, f) == (size_t)n); fclose(f);
    for (int i = 0; i < n; i++) {
        for (int s = 0; s <= SP; s++) for (int j = 0; j <= CH; j++) {
            float u = (float)s / SP, xc = (float)j / CH;
            v3 l = wing_pt(&gs[i], -u, xc, 0, 1), r = wing_pt(&gs[i], u, xc, 0, 1);
            assert(fabsf(l.x-r.x) < 1e-6f && fabsf(l.y+r.y) < 1e-6f && fabsf(l.z-r.z) < 1e-6f);
        }
        v3 V[MAX_V]; int T[MAX_T][3]; MeshOut mo;
        build_mesh(&gs[i], V, T, &mo);
        assert(mo.nv > 0 && mo.nv <= MAX_V && mo.nt > 0 && mo.nt <= MAX_T);
        assert(fits_print_volume(V, mo.nv, PRINT_VOLUME_M - PLA_WALL_M));
        MassProps mp; mass_props(&gs[i], V, T, mo.nt, &mp);
        assert(mp.mass > 0.005f && mp.mass < 0.100f);
        assert(mp.I[0] > 0 && mp.I[1] > 0 && mp.I[2] > 0);
        ScenarioDesc d = scenario_desc(1000 + i);
        v3 shift; MassProps printed = printed_mass_props(mp, d, &shift);
        assert(printed.mass > mp.mass);
        assert(printed.invmass > 0 && printed.invI[0] > 0 && printed.invI[4] > 0 && printed.invI[8] > 0);
        assert(len(shift) < 0.02f);
    }
    std::filesystem::remove_all(dir);
    puts("host geometry tests passed");
    return 0;
}
