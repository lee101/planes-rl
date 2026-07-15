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

clean:
	rm -f planes planes.exe

.PHONY: clean evolve evaluate videos robust-video stl presets host-test


presets: planes$(EXE)
	./planes$(EXE) presets --out out/presets --th 0.45


host-test:
	g++ -O2 -std=c++17 -x c++ src/stl.cu src/designs.cu tests/geometry_test.cpp -o out/geometry-test
	./out/geometry-test
