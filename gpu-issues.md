# GPU reliability incident report — RTX 5090 falling off PCIe bus (Xid 79)

Prepared 2026-07-15 to hand to hosting provider. Two hard GPU failures within 40 minutes, both under sustained compute load. The second occurred 14 minutes after a fresh reboot, so this is not OS-state accumulation.

## System

| | |
|---|---|
| Chassis/board | Supermicro SYS-6028R-T (X10DRi, dual-socket, 2U), BIOS 3.4a (2021-08-16) |
| GPU | Gigabyte GeForce RTX 5090 32GB (10de:2b85, subsys 1458:416f), physical slot 4, NUMA node 1, PCIe addr 0000:82:00.0 |
| Driver | nvidia 595.71.05 (open kernel modules, DKMS) — newest available in Ubuntu 24.04 repos |
| Kernel | 6.8.0-134-generic, Ubuntu 24.04 |

## Timeline (UTC, 2026-07-15)

- ~03:37 — Xid 79 "GPU has fallen off the bus" during heavy CUDA load (GA sim + vLLM inference resident). nvidia-smi dead. Module reload + PCI remove/rescan did NOT recover it.
- 03:57 — automated recovery rebooted the node.
- 04:02 — node up, GPU healthy.
- 04:16:35 — Xid 79 again, 14 minutes after boot, again under sustained compute (8192-plane CUDA GA at full SM occupancy alongside vLLM). Followed immediately by **Xid 154: "GPU recovery action changed to Node Reboot Required"** — driver states only a node reboot recovers.
- After the drop, config space reads return `!!! Unknown header type 7f` (device unresponsive on the bus); PCI remove/rescan re-enumerates the device but the driver cannot initialize it.

## Evidence

```
NVRM: Xid (PCI:0000:82:00): 79, GPU has fallen off the bus.
NVRM: Xid (PCI:0000:82:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
lspci -vv -s 82:00.0 -> !!! Unknown header type 7f
```

Diagnostics snapshots retained at `/var/lib/gpu-recover/diag-20260715-*.log` on the host.

## Analysis

Xid 79 with config-space 0x7f reads is a link-level/hardware event, not a software crash. Two occurrences under load, one straight after a clean boot, point at (in order of likelihood):

1. **Power delivery transients.** The RTX 5090 has a 575W board limit with documented sub-millisecond spikes toward ~900W on the 12V-2x6 connector. In a 2014-era 2U server chassis this typically runs through adapter cabling from PSU distribution never designed for that transient profile. Voltage droop on a spike drops the PCIe link → "fallen off the bus."
2. **PCIe signal integrity.** A Gen5 card negotiating in a Gen3 X10DRi slot, likely via riser in a 2U chassis. Marginal riser/slot seating produces exactly this failure signature under load (higher link activity + heat + mechanical/thermal expansion).
3. **Thermals.** A 3.5-slot consumer axial-fan card inside a 2U server has severely restricted airflow; VRM/hotspot excursions can also drop the link.

## What we (tenant) have already done on the host

- **Power cap at boot**: systemd unit `nvidia-powercap.service` runs `nvidia-smi -pm 1 && nvidia-smi -pl 400` (400W, down from 575W) — flattens the transient spikes that were the most likely trigger.
- **Disabled PCIe ASPM**: `pcie_aspm=off` added to kernel cmdline (L-state entry/exit on old chipsets with new GPUs is a known Xid 79 trigger).
- Driver already at the newest available (595.71.05-open); no newer release in Ubuntu 24.04 repos.
- Automated recovery ladder (`/usr/local/bin/gpu-recover`) in place: escalates smi-check → reset → module reload → PCI rescan → guarded reboot; snapshots diagnostics per event.
- Our compute jobs use short preemptible kernel launches; we can additionally lock clocks (`nvidia-smi -lgc`) if drops recur at 400W.

## Requests for the hosting provider

1. **Reseat the GPU and inspect/replace the PCIe riser** (slot 4, addr 82:00.0). Please check for connector wear and confirm the card is mechanically supported (3.5-slot card sag in a 2U will stress the slot).
2. **Inspect the 12V-2x6 / PCIe power cabling and adapters** feeding the card. If it's on Y-splitters or daisy-chained PCIe 8-pin → 12V-2x6 adapters from the PSU backplane, please give it dedicated runs. Confirm PSU model + rail capacity is adequate for a 575W-class card with ~900W transients.
3. **Check chassis airflow** around the card; confirm intake to the GPU fans isn't blocked and consider raising chassis fan floor.
4. **Update BIOS** if a newer X10DRi release than 3.4a exists (PCIe stability fixes), and verify slot 4 is set to Gen3 (not auto-negotiating unstably), Above-4G decoding on.
5. If drops persist after the above: **swap the card to a different slot/riser**, and if it still follows the card, RMA the GPU.

## How to reproduce

Sustained full-occupancy CUDA compute (any large GEMM burn or our GA sim: `github.com/lee101/planes-rl`, `make evolve`) alongside a resident vLLM instance reliably produced the drop within ~15–40 min at the 575W default limit. With the 400W cap we will report whether it recurs.

## Current state

GPU is down pending a node reboot (Xid 154). Automated reboot is cooldown-gated until ~09:57 UTC; manual reboot restores it sooner.
