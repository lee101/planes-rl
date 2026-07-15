NVCC ?= nvcc
ARCH ?= -arch=sm_120
FLAGS = -O3 -use_fast_math -std=c++17 $(ARCH) -Xcompiler -fopenmp -lineinfo

SRC = src/main.cu src/sim.cu src/render.cu src/stl.cu

planes: $(SRC) src/common.h src/sim.h
	$(NVCC) $(FLAGS) $(SRC) -o planes

LOCK = flock /var/lock/gpu-heavy.lock

evolve: planes
	$(LOCK) ./planes evolve --pop 8192 --gens 3000 --seeds 3 --out out

videos: planes
	$(LOCK) ./planes showcase --pop-file out/pop.bin --n 300 --sec 12 --o out/showcase.mp4 --label "planes-rl - 300 evolved paper planes"
	$(LOCK) ./planes hero --best out/best.bin --sec 10 --o out/hero.mp4 --label "planes-rl - best design chase cam"
	$(LOCK) ./planes turntable --best out/best.bin --sec 8 --o out/turntable.mp4 --label "planes-rl - evolved design turntable"

stl: planes
	./planes stl --best out/best.bin --o out/plane.stl --th 0.6

clean:
	rm -f planes

.PHONY: clean evolve videos stl
