#pragma once
#include "common.h"

struct float8 { float a, b, c, d, e, f, g, h; };

void run_ga(int pop, int gens, int nseeds, const char* out_dir, const char* init_path);
void evaluate_design(const Genome* g, int scenarios, int base_seed, const char* csv_path);
void select_finalist(const Genome* gs, int count, int scenarios, int base_seed, const char* out_dir);
void* panelbuf_alloc(int pop);
void panelbuf_free(void* pb);
void gpu_prepare(const Genome* d_gs, void* pb, int pop);
void gpu_sim(void* pb, int pop, int seed, float* d_fitness,
             float8* d_rec, int rec_every, int rec_cap, float spread, int vary_scenarios);
