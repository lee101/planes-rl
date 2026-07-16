#include "common.h"
#include <stdio.h>
#include <filesystem>
#include <vector>

void export_stl(const Genome*, const char*, float);

struct NamedDesign { const char* name; Genome g; };

static Genome design(float span, float chord, float taper, float sweep,
                     float di, float dout, float kink, float c1, float c2,
                     float washout, float elevator, float nose, float keel,
                     float gap, float deck_scale, float body,
                     float fin_h, float fin_chord, float fin_sweep,
                     float tail_span, float tail_chord, float tail_angle, float style) {
    Genome g = { span, chord, taper, sweep, di, dout, kink, c1, c2,
                 washout, elevator, nose, keel, gap, deck_scale, body,
                 fin_h, fin_chord, fin_sweep, tail_span, tail_chord, tail_angle };
    g.wing_le_b1 = -0.45f + 0.75f*style;
    g.wing_le_b2 =  0.35f - 0.65f*style;
    g.chord_b1 = 0.30f - 0.45f*style;
    g.chord_b2 = -0.15f + 0.55f*style;
    g.fin_layout = style;
    g.fin_cant = 0.12f + 0.45f*style;
    g.fin_taper = 0.30f + 0.55f*(1.0f-style);
    g.fin_span_pos = 0.35f + 0.55f*style;
    g.tail_sweep = -0.10f + 0.75f*style;
    g.tail_taper = 0.95f - 0.55f*style;
    g.tail_dihedral = -0.12f + 0.30f*style;
    g.wing_tip_round = 0.15f + 0.75f*style;
    g.center_chord = -0.10f + 0.55f*(1.0f-style);
    return g;
}

static std::vector<NamedDesign> preset_designs() {
    return {
        { "stable_gull", design(.235f,.105f,.58f,.24f, .13f,.58f,.72f,.025f,.010f, -.055f,.035f,.42f,.027f,0,.8f,.12f, .038f,.35f,.30f,.36f,.22f,-.025f,.10f) },
        { "fast_dart", design(.205f,.145f,.16f,.52f, .06f,.30f,.78f,.012f,-.005f, -.035f,.025f,.72f,.023f,0,.75f,.10f, .022f,.22f,.60f,.18f,.14f,.015f,.30f) },
        { "efficient_plank", design(.245f,.082f,.82f,.12f, .10f,.42f,.68f,.030f,.004f, -.070f,.060f,.30f,.032f,0,.9f,.16f, .030f,.42f,.18f,.48f,.18f,-.045f,.55f) },
        { "boxy_biplane", design(.205f,.090f,.70f,.16f, .08f,.32f,.70f,.022f,.005f, -.040f,.035f,.38f,.024f,.034f,.82f,.14f, .045f,.48f,.22f,.42f,.28f,-.020f,.72f) },
        { "compact_trainer", design(.185f,.100f,.68f,.20f, .18f,.72f,.62f,.035f,.012f, -.065f,.050f,.48f,.036f,0,.8f,.18f, .050f,.55f,.12f,.58f,.32f,-.060f,.92f) }
    };
}

static void describe(const char* name, const Genome& g) {
    v3 V[MAX_V]; int T[MAX_T][3]; MeshOut mo;
    build_mesh(&g, V, T, &mo);
    MassProps mp; mass_props(&g, V, T, mo.nt, &mp);
    v3 e = mesh_extent(V, mo.nv);
    printf("%-18s %6.1fg  %3.0fx%3.0fx%3.0f mm  CG=(%5.1f,%5.1f,%5.1f) mm  %s\n",
           name, mp.mass * 1000, e.x*1000, e.y*1000, e.z*1000,
           mp.com.x*1000, mp.com.y*1000, mp.com.z*1000,
           fits_print_volume(V, mo.nv, PRINT_VOLUME_M - PLA_WALL_M) ? "printable" : "TOO LARGE");
}

void inspect_design(const Genome* g, const char* name) { describe(name, *g); }

void write_presets(const char* out_dir, float thickness_mm) {
    std::filesystem::create_directories(out_dir);
    auto ds = preset_designs();
    std::filesystem::path seed_path = std::filesystem::path(out_dir) / "seeds.bin";
    FILE* f = fopen(seed_path.string().c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", seed_path.string().c_str()); return; }
    int n = (int)ds.size(); fwrite(&n, sizeof n, 1, f);
    for (const auto& d : ds) fwrite(&d.g, sizeof d.g, 1, f);
    fclose(f);
    for (const auto& d : ds) {
        describe(d.name, d.g);
        std::filesystem::path stl = std::filesystem::path(out_dir) / (std::string(d.name) + ".stl");
        export_stl(&d.g, stl.string().c_str(), thickness_mm);
    }
    printf("preset seed population -> %s\n", seed_path.string().c_str());
}
