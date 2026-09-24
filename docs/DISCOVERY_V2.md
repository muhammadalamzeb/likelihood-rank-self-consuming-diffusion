# Discovery under Constraint Set v2

**Started:** 2026-09-23  
**Mode:** New Phase-1 landscape (not Reboot #15 under v1)  
**Status:** Landscape + candidate shortlist; **no topic lock**

## What changed vs v1

v1 exhausted text/NLP GenAI under L3–4-only lock rules. v2 opens modality space and allows strong L2 empirical papers, with larger compute when the core remains reproducible.

## Field map (crowded vs apparent openings)

| Subarea | 2025–2026 status | Apparent opening under v2 | Solo feasibility |
|--------|-------------------|---------------------------|------------------|
| Unified understanding+generation VLMs | Crowded surveys; tokenization & hybrid AR–diffusion open at foundation-model scale | Small open models: when joint training helps vs interferes | Medium–hard |
| Controllable T2I / ControlNet conflicts | SemanticControl, Cross-ControlNet, MIControlNet own **methods** | Possibly a **regime map / law** of conflict types (not another fixer) | High (inference-time) |
| Diffusion editing tradeoffs | Controllability–fidelity theory + ALE leakage | Thin — mechanisms largely claimed | Low novelty |
| Video physics & temporal compositionality | VideoPhy-2, VBench-2.0, TC-Bench | **Axis dissociation**: does physics score predict temporal compositionality? | Medium (eval-heavy L2) |
| Visual tokenizer ↔ AR generator | CRT, GigaTok, EOSTok, VTP | Possibly **small-compute flip** of compression–generation tradeoff | Medium |
| Codec–LM audio | LLM-Codec, NAACL co-design, SPG-Codec | Thin — objective mismatch claimed | Low novelty |
| Lip-sync leakage | HighSync, KeySync, SyncAnyone | Thin — visual shortcut mechanism claimed | Low novelty |
| Text-to-3D Janus / view bias | SEGS, ConsDreamer, TD-Attn, older debiasing | Thin at mechanism level | Low novelty |
| Diffusion training dynamics | SpeeD, adaptive timesteps, Data Warmup | **Interaction** of timestep sampling × data curriculum under small batch | High |
| Long-form music structure | MusicLayout, ACG, Yin-Yang, RPPNet | Possibly a **measurable structural-drift law** vs another planner | Medium |
| T2I automatic metrics | CROC, FineGRAIN, PROTOBIAS, construct-validity audits | Thin for new metric papers | Low novelty |
| VLM visual–linguistic arbitration | Arbitration-failure, PIH heads, confounder propagation | Thin — do not clone text UQ/hallucination kills into vision | Avoid |

## Carry-forward risk from v1

Do not reintroduce v1 clones of UQ / selective generation / RAG conflict / MDM-SER / etc. unless the scientific question is **materially** different under a new modality.

## Candidate IDs (for kill-test; not locked)

Round 1 **V2-1…V2-6 → all KILL**.  
Round 2 **V2-9…V2-13 → all KILL**.  
Round 3 **V2-14…V2-18 → all KILL**.  
Strategy B stacks v0–v2 observation clusters → **all KILL** (see `KILL_LOG.md` FC*/FM1/T3; exclusions through **38**).  
No Phase 4.
