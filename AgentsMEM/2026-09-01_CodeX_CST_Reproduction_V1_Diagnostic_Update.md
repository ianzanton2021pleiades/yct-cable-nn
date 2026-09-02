# CodeX memory update: CST_Reproduction V1 diagnostic

## Session context

- Agent: CodeX
- Date: 2026-09-01 Asia/Shanghai
- Workspace: `D:\GitRepository\Cable-NN`
- Parent memory: `AgentsMEM\2026-09-01_221701_CodeX_CST_Reproduction_Conversation_Memory.md`
- V0 is frozen by the user. `DifferentAgent\CodeX\CST_Reproduction\V1` is reserved for the corrected model; no V0 or V1 code was changed in this diagnostic turn.

## Verified findings

- `E:\FDR案例-csv\RG58缺陷制造实验` contains Health-1, Health-2, CutPVC-1 and guh ua, each 10000 points on the same 2.5 kHz–2 GHz grid. The trailing empty CSV column is cosmetic.
- With the exact current V0 FDR settings, Algorithm 1 and Algorithm 2 both localize the incremental defect near 49.2 m. CutPVC: 49.233 m (A1) and 49.162 m (A2 smoothed); guh ua: 49.183 m (A1) and 49.148 m (A2 smoothed).
- Common-scale differential impulse RMS ratio for 60–80 m post-region versus 45–60 m local region is A1 Cut .090, A1 guh ua .023, A2 Cut .183, A2 guh ua .029. This supports no dominant new post-defect impulse train in this setup; terminal 80–95 m must be treated separately. The strict scientific wording is “below the measurement/algorithm floor or health repeatability”, not “exactly zero”.
- Actual Health-1 Algorithm-2 output matches REF `compute_response` on the full trace: max errors step_raw `8.9e-16`, step_smoothed `0`, impulse_raw `3.2e-15`, impulse_smoothed `7.1e-17`.
- V0 controlled ablation on the 10000-point CST grid: current A1/A2 post-joint step residual in 15–20 m is `.007756/.2008`; deleting all internal joints reduces it to `5.77e-5/.006573`; changing 300-ohm transition lines to about the standard-line impedance reduces 20–25 m residual to `.003961/.09377`.
- V0 standard cell has `Z0≈136.93 ohm`, `VF≈.6090`, and discrete cutoff `fc≈145 MHz`, while the FDR scales are A1 `epsr=2.23` (VF≈.6694) and A2 `VF=.6715`; the port is 50 ohm and V0 line loss is only about `.00823 dB/m` (`.66 dB` round trip over 40 m).

## Interpretation and plan

- V0 oscillation is mainly a model/finite-band response: strong 50/136.9/135/300-ohm discontinuities, a 1-uH/8-pF special joint, insufficient loss and the 145-MHz ladder cutoff. Hann sidelobes and A2 smoothing broaden it but are not the primary bug.
- A1 and A2 showing the same V0 event positions is evidence against an Algorithm-2-only implementation fault.
- Keep V0 as the exact report/CST reference. V1 should expose model audit switches and compare continuous fixed RLGC versus a properly scaled finer ladder.
- DG V3 frequency-dependent loss and Field/RG58 correction are explicitly deferred; this V1 task only improves the existing CST-equivalent program.
- Durable plan document: `AgentsDoc\CST_Reproduction_V1_问题定位与实施计划.md`.

## Locked follow-up scope

V1 targets a numerically improved CST-equivalent program retaining the report's 136.9/135/300-ohm parameters, with a 0.1 m default ladder and continuous fixed-RLGC/reference ablations. Field/RG58 fitting and DG V3 frequency-dependent loss are future tasks, not part of this V1 implementation.
