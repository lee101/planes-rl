NVCC ?= nvcc
ARCH ?= -arch=native
FLAGS = -O3 -use_fast_math -std=c++17 $(ARCH) -lineinfo

SRC = src/main.cu src/sim.cu src/render.cu src/stl.cu src/designs.cu

ifeq ($(OS),Windows_NT)
EXE = .exe
LOCK =
else
EXE =
LOCK = flock /var/lock/gpu-heavy.lock
endif

planes$(EXE): $(SRC) src/common.h src/sim.h
	$(NVCC) $(FLAGS) $(SRC) -o planes$(EXE)


evolve: planes$(EXE)
	$(LOCK) ./planes$(EXE) evolve --pop 8192 --gens 3000 --seeds 8 --out out

evaluate: planes$(EXE)
	$(LOCK) ./planes$(EXE) evaluate --best out/best.bin --n 512 --seed 9001 --o out/robustness.csv

videos: planes$(EXE)
	$(LOCK) ./planes$(EXE) showcase --pop-file out/pop.bin --n 300 --sec 12 --o out/showcase.mp4 --label "planes-rl - 300 evolved printable PLA gliders"
	$(LOCK) ./planes$(EXE) hero --best out/best.bin --sec 10 --o out/hero.mp4 --label "planes-rl - best design chase cam"
	$(LOCK) ./planes$(EXE) turntable --best out/best.bin --sec 8 --o out/turntable.mp4 --label "planes-rl - evolved design turntable"

robust-video: planes$(EXE)
	$(LOCK) ./planes$(EXE) robust-video --best out/best.bin --n 96 --sec 12 --o out/robust.mp4 --label "planes-rl - one glider across 96 wind and launch scenarios"

stl: planes$(EXE)
	./planes$(EXE) stl --best out/best.bin --o out/plane.stl --th 0.45

realistic-run: planes$(EXE) presets
	$(LOCK) ./planes$(EXE) evolve --pop 10000 --gens 300 --seeds 8 --out out/realistic-run --init out/presets/seeds.bin
	$(LOCK) ./planes$(EXE) evaluate --best out/realistic-run/best.bin --n 10000 --seed 730001 --o out/realistic-run/robustness-10000.csv
	./planes$(EXE) stl --best out/realistic-run/best.bin --o out/realistic-run/best-plane.stl --th 0.45
	$(LOCK) ./planes$(EXE) showcase --pop-file out/realistic-run/archive.bin --n 300 --sec 12 --o out/realistic-run/evolution-300x10000.mp4 --label "300 generations x 10000 PLA gliders - print and launch robust"

realistic-run2: planes$(EXE)
	$(LOCK) ./planes$(EXE) evolve --pop 100000 --gens 100 --seeds 8 --out out/realistic-run2 --init out/realistic-run/pop.bin
	$(LOCK) ./planes$(EXE) evaluate --best out/realistic-run2/best.bin --n 10000 --seed 930001 --o out/realistic-run2/robustness-10000.csv
	./planes$(EXE) stl --best out/realistic-run2/best.bin --o out/realistic-run2/best-plane.stl --th 0.45
	$(LOCK) ./planes$(EXE) evolution-video --pop-file out/realistic-run2/archive.bin --n 100 --sec 20 --o out/realistic-run2/evolution-100x100000-twitter.mp4 --label "100 chronological generations x 100000 PLA gliders"

shape-robust-run: planes$(EXE) presets
	$(LOCK) ./planes$(EXE) evolve --pop 100000 --gens 300 --seeds 12 --out out/shape-robust-run --init out/presets/seeds.bin
	$(LOCK) ./planes$(EXE) evaluate --best out/shape-robust-run/best.bin --n 100000 --seed 2930001 --o out/shape-robust-run/final-robustness-100000.csv
	./planes$(EXE) stl --best out/shape-robust-run/best.bin --o out/shape-robust-run/best-final-plane.stl --th 0.45

clean:
	rm -f planes planes.exe

.PHONY: clean evolve evaluate videos robust-video stl presets host-test realistic-run realistic-run2 shape-robust-run


presets: planes$(EXE)
	./planes$(EXE) presets --out out/presets --th 0.45


host-test:
	g++ -O2 -std=c++17 -x c++ src/stl.cu src/designs.cu tests/geometry_test.cpp -o out/geometry-test
	./out/geometry-test
