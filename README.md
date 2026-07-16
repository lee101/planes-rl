# planes-rl

CUDA-accelerated evolutionary design of hand-thrown, 3D-printable gliders. The optimizer generates aircraft, computes PLA mass properties, evaluates thousands of noisy 6DOF launches in parallel, and exports printable STL shells plus GPU-rendered videos.

The optimizer is a genetic algorithm (rather than a policy-learning RL agent), but it performs the requested design search over a randomized flight environment.

## Current design target

- Maximum build envelope: **250 x 250 x 250 mm**.
- Material: ordinary PLA at **1240 kg/m3**.
- Optimization wall: **0.45 mm** single-wall shell (`PLA_WALL_M`).
- Typical included designs: roughly 16-25 g including parameterized nose ballast and empennage.
- Hand launch domain: 7.2-11.2 m/s, varied held attitude, independent force-direction error, release height/rates, off-CG grip impulse, calm-to-severe gust fields, head/tailwind, crosswind, and air density.
- Print domain: PLA density/extrusion scale, 6-40% equivalent infill/rib mass, asymmetric infill/adhesion CG bias, layer-dependent skin drag, and mild pitch/roll warping.
- Tail-aware objective: 65% mean distance, 25% worst-quartile CVaR, and 10% single worst case. Oversize designs receive a hard negative fitness.
- Champion selection tests each generation's top eight candidates on an independent 128-scenario GPU holdout suite, preventing lucky training specialists from becoming `best.bin`.

## Geometry

A 35-float genome controls the original wing, camber, ballast, keel, and optional biplane parameters plus cubic-Bezier leading-edge and chord distributions, elliptical tip blending, center-chord expansion, swept/tapered/dihedral horizontal tails, and center/twin/triple/outboard fin topology with cant, taper, and span position. Every planform and paired fin is mirrored exactly about the center plane as a structural symmetry heuristic. The expanded search can express curved, elliptical, plank, swept, and strongly triangular/delta families without introducing accidental left/right imbalance.

`mass_props()` treats every aerodynamic triangle as a uniform extruded PLA lamina. It integrates triangle second moments, includes through-thickness inertia, adds the nose ballast, then computes center of mass and the full inverse inertia tensor. STL mass reporting scales with the requested export thickness.

## Robust GPU flight simulation

Each CUDA block owns one plane. Threads stride its panels, calculate local relative flow including rotational velocity and print warp, reduce force/torque with warp shuffles, and integrate a 6DOF rigid body at 600 Hz. Kernels run in one-second chunks to avoid long uninterruptible launches.

The release is impulse-based. Aircraft attitude and force direction are randomized separately, and the impulse is applied at a randomized grip point. Its moment about the print-adjusted CG is converted through the full inertia tensor into release rotation. Print variation changes mass, CG, inertia, drag, and local surface normals for each scenario.

For each evaluation seed, every candidate receives the same deterministic imperfect throw, preserving fair rankings. Across seeds the optimizer sees different:

- launch speed and release height;
- pitch, yaw, and roll errors;
- pitch/roll/yaw release rates;
- steady wind and multi-octave gust fields.

Use at least 8 seeds for serious robust optimization.

## GPU renderer

The renderer is CUDA-only: procedural sky/ground, shadows, flight ribbons, wind particles, and a packed 64-bit atomic depth/color buffer streamed to `ffmpeg`. Swarm rendering uses exact per-plane triangle prefix ranges instead of rasterizing `MAX_T` empty slots for every aircraft, and frame readback uses pinned host memory. Video output is 1280x720 at 24 fps, H.264/yuv420p, fast-start MP4, with an 8 Mbps ceiling. `evolution-video` interpolates archive champions in chronological order so geometry changes remain readable.

## Build

Requires CUDA, a compatible NVIDIA GPU, a C++17 host compiler, and `ffmpeg`.

```bash
make                         # uses -arch=native for the installed GPU
./planes                     # print CLI help
```

Override `ARCH` for another GPU, for example `sm_89` on Ada.

## Quick start

Generate five printable baseline designs and a warm-start population:

```bash
make presets
# out/presets/stable_gull.stl
# out/presets/fast_dart.stl
# out/presets/efficient_plank.stl
# out/presets/boxy_biplane.stl
# out/presets/compact_trainer.stl
# out/presets/seeds.bin
```

Run a small smoke optimization:

```bash
./planes evolve --pop 1024 --gens 50 --seeds 8 --out out/run1 \
  --init out/presets/seeds.bin
```

Run the full search:

```bash
./planes evolve --pop 8192 --gens 3000 --seeds 8 --out out \
  --init out/presets/seeds.bin
```

Reproduce the realistic 300-generation, 10,000-aircraft run and its artifacts:

```bash
make realistic-run
# out/realistic-run/best-plane.stl
# out/realistic-run/evolution-300x10000.mp4
# out/realistic-run/robustness-10000.csv
```

Run the larger 100-generation search with 100,000 aircraft per generation and
produce a chronological, social-media-ready evolution reel:

```bash
make realistic-run2
# out/realistic-run2/best-plane.stl
# out/realistic-run2/evolution-100x100000-twitter.mp4
```

Run the high-diversity Bezier/multi-fin search without spending GPU time on video:

```bash
make shape-robust-run
# 300 generations x 100,000 geometries x 12 training environments
# out/shape-robust-run/best-final-plane.stl
# out/shape-robust-run/final-robustness-100000.csv
```

Audit the champion across thousands of independent environments, then inspect and export it:

```bash
./planes evaluate --best out/best.bin --n 4096 --seed 9001 \
  --o out/robustness-4096.csv
./planes inspect --best out/best.bin
./planes stl --best out/best.bin --o out/plane.stl --th 0.45
```

An STL export is refused if any mesh dimension exceeds 250 mm. Use the same wall thickness used by optimization unless deliberately testing a heavier build.

Render results, including one champion launched through 96 independent hostile environments with long trajectory trails:

```bash
./planes showcase --pop-file out/pop.bin --n 300 --sec 12 \
  --o out/showcase.mp4 --label "planes-rl - robust PLA gliders"
./planes hero --best out/best.bin --sec 10 --o out/hero.mp4
./planes robust-video --best out/best.bin --n 96 --sec 12 \
  --o out/robust.mp4 --label "one design - 96 winds and imperfect throws"
./planes turntable --best out/best.bin --sec 8 --o out/turntable.mp4
```

`archive.bin` has one generation champion per entry and can also be passed to `showcase` to visualize how the design family changes during optimization.

## Outputs

- `best.bin`: best robust genome seen.
- `pop.bin`: final population sorted by evaluation score.
- `archive.bin`: generation champions, suitable for an evolution showcase.
- `history.csv`: training-best, independent 128-scenario holdout, population mean, and top-decile score by generation.
- `robustness-*.csv`: per-scenario distance plus launch, density, drag, gust, and base-wind telemetry.
- STL and MP4 outputs selected through the CLI.

## External reference aircraft

See `REFERENCE_MODELS.md` for downloadable community gliders worth comparing against. They are linked rather than vendored so their individual licenses and attribution requirements remain explicit. Arbitrary STL geometry is not yet converted into the 16-gene search space; use reference models for physical baselines, or fit their proportions into a preset genome.

## Important physical limitations

This is a fast panel model, not CFD or a slicer. Infill is an equivalent distributed-mass model with randomized imbalance; the STL itself does not encode a slicer's infill path. The simulator does not yet resolve layer-line strength/impact breakage, nozzle corner rounding, support scars, detailed internal toolpaths, or aeroelastic wing bending. Validate finalists with a slicer and real throws, then calibrate the modeled distributions using measured mass, CG, release, and wind data.

## Windows toolchain

Install Visual Studio 2022 Build Tools with the C++ workload, CUDA Toolkit, GNU Make, and ffmpeg. The Makefile uses `-arch=native`, emits `planes.exe`, avoids Linux-only `flock`, and the renderer maps `popen` to `_popen`. Run `./build-windows.ps1`; it discovers Visual Studio Build Tools, initializes the MSVC environment, finds the Scoop `nvcc` shim, and compiles for the local RTX 3070 (`sm_86`). Pass `-Arch sm_120` on the RTX 5090 host.
