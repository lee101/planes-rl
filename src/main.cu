#include "common.h"
#include "sim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

void render_showcase(const Genome*, int n, const char* out, int seconds, const char* label);
void render_hero(const Genome*, const char* out, int seconds, const char* label);
void render_robust(const Genome*, int scenarios, const char* out, int seconds, const char* label);
void render_turntable(const Genome*, const char* out, int seconds, const char* label);
void render_evolution(const Genome*, int n, const char* out, int seconds, const char* label);
void export_stl(const Genome*, const char* path, float thickness_mm);
void write_presets(const char* out_dir, float thickness_mm);
void inspect_design(const Genome*, const char* name);

static const char* opt(int argc, char** argv, const char* key, const char* dflt) {
    for (int i = 2; i < argc - 1; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
    return dflt;
}

static Genome load_best(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p); exit(1); }
    Genome g; fread(&g, sizeof g, 1, f); fclose(f); return g;
}

static std::vector<Genome> load_pop(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p); exit(1); }
    int n; fread(&n, 4, 1, f);
    std::vector<Genome> gs(n);
    fread(gs.data(), sizeof(Genome), n, f); fclose(f); return gs;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: planes <evolve|evaluate|select-best|showcase|hero|robust-video|evolution-video|turntable|stl|inspect|presets> [opts]\n"
               "  evolve    --pop 4096 --gens 400 --seeds 8 --out out [--init out/pop.bin]\n"
               "  evaluate  --best out/best.bin --n 512 --seed 9001 --o out/robustness.csv\n"
               "  select-best --pop-file out/archive.bin --n 8192 --seed 1900001 --out out/run\n"
               "  showcase  --pop-file out/pop.bin --n 300 --sec 12 --o out/showcase.mp4 --label text\n"
               "  hero      --best out/best.bin --sec 10 --o out/hero.mp4 --label text\n"
               "  robust-video --best out/best.bin --n 96 --sec 12 --o out/robust.mp4\n"
               "  evolution-video --pop-file out/archive.bin --n 100 --sec 20 --o out/evolution.mp4\n"
               "  turntable --best out/best.bin --sec 8 --o out/turntable.mp4 --label text\n"
               "  stl       --best out/best.bin --o out/plane.stl --th 0.45\n"
               "  inspect   --best out/best.bin\n"
               "  presets   --out out/presets --th 0.45\n");
        return 1;
    }
    const char* mode = argv[1];
    if (!strcmp(mode, "evolve")) {
        run_ga(atoi(opt(argc, argv, "--pop", "4096")), atoi(opt(argc, argv, "--gens", "400")),
               atoi(opt(argc, argv, "--seeds", "3")), opt(argc, argv, "--out", "out"),
               opt(argc, argv, "--init", nullptr));
    } else if (!strcmp(mode, "evaluate")) {
        Genome g = load_best(opt(argc, argv, "--best", "out/best.bin"));
        evaluate_design(&g, atoi(opt(argc, argv, "--n", "512")),
                        atoi(opt(argc, argv, "--seed", "9001")),
                        opt(argc, argv, "--o", "out/robustness.csv"));
    } else if (!strcmp(mode, "select-best")) {
        auto gs = load_pop(opt(argc, argv, "--pop-file", "out/archive.bin"));
        select_finalist(gs.data(), (int)gs.size(), atoi(opt(argc, argv, "--n", "8192")),
                        atoi(opt(argc, argv, "--seed", "1900001")), opt(argc, argv, "--out", "out"));
    } else if (!strcmp(mode, "showcase")) {
        auto gs = load_pop(opt(argc, argv, "--pop-file", "out/pop.bin"));
        int n = atoi(opt(argc, argv, "--n", "300"));
        if ((int)gs.size() < n) n = gs.size();
        render_showcase(gs.data(), n, opt(argc, argv, "--o", "out/showcase.mp4"),
                        atoi(opt(argc, argv, "--sec", "12")), opt(argc, argv, "--label", "planes-rl"));
    } else if (!strcmp(mode, "hero")) {
        Genome g = load_best(opt(argc, argv, "--best", "out/best.bin"));
        render_hero(&g, opt(argc, argv, "--o", "out/hero.mp4"),
                    atoi(opt(argc, argv, "--sec", "10")), opt(argc, argv, "--label", "planes-rl hero"));
    } else if (!strcmp(mode, "robust-video")) {
        Genome g = load_best(opt(argc, argv, "--best", "out/best.bin"));
        render_robust(&g, atoi(opt(argc, argv, "--n", "96")),
                      opt(argc, argv, "--o", "out/robust.mp4"),
                      atoi(opt(argc, argv, "--sec", "12")),
                      opt(argc, argv, "--label", "planes-rl - one design, 96 hostile environments"));
    } else if (!strcmp(mode, "turntable")) {
        Genome g = load_best(opt(argc, argv, "--best", "out/best.bin"));
        render_turntable(&g, opt(argc, argv, "--o", "out/turntable.mp4"),
                         atoi(opt(argc, argv, "--sec", "8")), opt(argc, argv, "--label", "planes-rl best design"));
    } else if (!strcmp(mode, "evolution-video")) {
        auto gs = load_pop(opt(argc, argv, "--pop-file", "out/archive.bin"));
        int n = atoi(opt(argc, argv, "--n", "100"));
        if ((int)gs.size() < n) n = gs.size();
        render_evolution(gs.data(), n, opt(argc, argv, "--o", "out/evolution.mp4"),
                         atoi(opt(argc, argv, "--sec", "20")),
                         opt(argc, argv, "--label", "chronological evolution - one champion per generation"));
    } else if (!strcmp(mode, "stl")) {
        Genome g = load_best(opt(argc, argv, "--best", "out/best.bin"));
        export_stl(&g, opt(argc, argv, "--o", "out/plane.stl"), atof(opt(argc, argv, "--th", "0.45")));
    } else if (!strcmp(mode, "inspect")) {
        Genome g = load_best(opt(argc, argv, "--best", "out/best.bin"));
        inspect_design(&g, opt(argc, argv, "--best", "out/best.bin"));
    } else if (!strcmp(mode, "presets")) {
        write_presets(opt(argc, argv, "--out", "out/presets"), atof(opt(argc, argv, "--th", "0.45")));
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 1;
    }
    return 0;
}
