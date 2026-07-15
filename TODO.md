# TODO

## Status (2026-07-15)
- Code builds and runs on sm_120. Pushed to lee101/planes-rl.
- Fixed: `k_prepare`/`k_meshgen` rewritten one-block-per-plane with shared-memory mesh (per-thread `V[512]`/`T[900][3]` local arrays reserved ~4GB device local mem -> OOM alongside vLLM).
- Added: GA checkpoints every 25 gens (best/pop/archive) + `--init out/pop.bin` warm start.
- Blocker: RTX 5090 fell off the bus (Xid 79) twice under sustained GA load (03:37, 04:16). `gpu-recover` ladder still restoring the driver. Existing out/history.csv is a partial 110-gen run (best ~12.6m).

## Next
1. Once GPU is back: `sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 400` (power cap to stop bus-drop transients).
2. `make evolve` (pop 8192, 3000 gens, 3 seeds; checkpoints survive crashes; resume with `--init out/pop.bin`).
3. `make videos && make stl`, move outputs to `media/` (showcase.mp4, hero.mp4, turntable.mp4, plane.stl).
4. Update README Results with measured best distance + final genome, commit media, push.
5. Optional: pin GPU clocks (`nvidia-smi -lgc`) if Xid 79 recurs at 400W; consider inter-chunk sleep throttle in `gpu_sim`.
