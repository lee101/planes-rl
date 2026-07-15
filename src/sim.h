#pragma once
#include "common.h"

struct float8 { float a, b, c, d, e, f, g, h; };

void run_ga(int pop, int gens, int nseeds, const char* out_dir, const char* init_path);
void* panelbuf_alloc(int pop);
void gpu_prepare(const Genome* d_gs, void* pb, int pop);
void gpu_sim(void* pb, int pop, int seed, float* d_fitness,
             float8* d_rec, int rec_every, int rec_cap, float spread);
