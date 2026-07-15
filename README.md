# planes-rl

GPU-accelerated evolution of paper plane designs for maximum flight distance. Pure C++/CUDA: a parametric plane generator, a 6DOF panel-method flight simulator, a genetic algorithm, and a CUDA software rasterizer that renders the whole swarm to video — everything runs on the GPU (built for an RTX 5090, sm_120).

https://github.com/lee101/planes-rl/raw/main/media/showcase.mp4

- `media/showcase.mp4` — 300 evolved planes launched into gusty wind, trajectory ribbons + wind streak particles
- `media/hero.mp4` — chase cam on the best design
- `media/turntable.mp4` — turntable of the champion design
- `media/plane.stl` — downloadable solid-shell STL of the champion (mm units)

## How it works

### Parametric plane genome (16 genes)

Every plane is generated from a 16-float genome (`src/common.h`): wingspan, root chord, taper, leading-edge sweep, two-segment dihedral (inner panel + winglet angle with a movable kink station), **cubic-bezier chordwise camber** (two control points + trailing-edge reflex/elevator), spanwise washout twist, nose ballast mass, keel depth, center body width, and **biplane genes** (deck gap + upper-deck scale — the optimizer is free to discover double-decker designs). The genome maps to a triangle mesh (~170-340 panels): wing surface grid with piecewise dihedral polyline, folded keel sheet, and optional second deck with struts.

Mass properties are computed from the mesh as a thin shell (80 gsm paper) plus point ballast at the nose; center of mass, inertia tensor, and its inverse are derived per design on-device.

### Flight simulation (CUDA, one block per plane)

6DOF rigid body integration (semi-implicit Euler, dt = 1/600 s, quaternion attitude) with per-panel aerodynamics:

- Per panel: relative velocity `v_rel = v + ω×r − wind(x, t)`, signed inflow `s = v̂·n`
- Normal force blends attached thin-airfoil flow with Newtonian flat-plate stall:
  `C_N = 1.9·s·√(1−s²) + 2·s·|s|` — gives realistic lift slope at small angles of attack and post-stall behavior, orientation-free (paper sheets fly either side up)
- Skin friction on the tangential component
- Rotational damping falls out naturally because each panel sees `ω×r`
- Wind field: light headwind/crosswind base + 3-octave sinusoidal gusts in space and time, seeded per evaluation

Kernel layout: one thread block per plane, 128 threads striding panels, warp-shuffle + shared-memory reduction of force/torque, thread 0 integrates. State stays in shared memory for a whole chunk (600 steps ≈ 1 s of sim per launch — kernels stay short and preemptible, which also keeps the driver/GSP happy). Fitness = horizontal distance when the plane hits the ground, averaged over 3 wind seeds per generation to force robust designs, launched at 9.5 m/s from 1.8 m.

Throughput: a full generation (8192 planes × 3 winds × up to 7200 steps × ~200 panels) evaluates in a fraction of a second on a 5090 — planes fan out across all SMs and dead planes retire their blocks early.

### Genetic algorithm

Population 8192, tournament-4 selection, blend crossover (extrapolating lerp), annealed per-gene gaussian mutation, elitism, 5% random immigrants per generation, rotating wind seeds each generation so nothing overfits one gust pattern. Best-of-generation archive is kept for the showcase scene.

### Visualization (CUDA software rasterizer)

No OpenGL — the renderer is also CUDA: procedural sky + checkered ground with 5 m distance lines computed per pixel by ray cast, then one thread per triangle rasterizes with perspective-correct depth into a 64-bit atomicMin z-buffer (depth in the high word, RGBA in the low word — depth test and write are a single atomic). The scene draws:

- every plane's mesh, posed from recorded sim states (60 fps pose capture), lambert-shaded, HSV-ranked colors
- projected ground shadows
- fading trajectory ribbons (camera-facing quad strips)
- 6000 wind streak particles advected through the same wind field the planes feel

Frames stream straight from the GPU into an `ffmpeg` rawvideo pipe → H.264. The turntable mode orbits the champion mesh; STL export offsets the sheet ±0.3 mm along vertex normals and stitches boundary edges into a watertight-ish solid shell for printing.

## Build + run

```bash
make            # nvcc -O3 -use_fast_math -arch=sm_120
make evolve     # GA: pop 8192, 3000 generations, 3 wind seeds -> out/best.bin, out/pop.bin, out/history.csv
make videos     # showcase (300 planes), hero chase cam, turntable
make stl        # out/plane.stl
```

CLI details: `./planes` with no args prints all modes/options.

## 5090 / performance notes

- FP32 state + `-use_fast_math`; aero inner loop is FMA-dense and register-resident, panels read via coalesced `float4` SoA
- Warp shuffle reductions, shared-memory block reduce, zero global traffic inside a timestep except panel reads
- Mesh building, mass properties, 3×3 inertia inversion all happen in device kernels — genomes are the only host↔device traffic in the GA loop (32 KB per generation up, 32 KB fitness down)
- Short chunked kernel launches instead of one mega-kernel per flight: same throughput, far better citizen on a shared/oversubscribed GPU
- FP4/FP16: deliberately not used for physics state — 6DOF integration over 7200 steps is numerically unforgiving and the sim is compute-bound on FMAs, not bandwidth; quantization would buy nothing here. The rasterizer's packed 64-bit z-buffer atomics are the bandwidth-bound part and already 8-byte packed.

## Results

See `out/history.csv` for the fitness curve and the release/media files for the evolved champion. Genome + measured distances are printed at the end of `make evolve`.
