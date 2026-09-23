# Kill log — adversarial novelty archive

Factual archive of candidates rejected during the topic-search phase.  
**Not** a literature review paper. Paper IDs are those used during kill-tests; verify before citing.

Kill reason codes: `NAMED` | `FRAMEWORK` | `ABLATION` | `FUTURE_WORK` | `CLEANER_SYNTH` | `COMBO` | `CLONE` | `CROWDED` | `EXCLUDED_NBHD`

---

## Round 0 — Initial candidates

| ID | Topic (short) | Decision | Reason | Closest anchors (examples) |
|----|---------------|----------|--------|----------------------------|
| C1 | PEFT/UQ drift + recalibration | KILL | NAMED / CROWDED | UQ4CT, TAD, semantic entropy, SAR |
| C1-lite | Diagnostic-only remnant | ARCHIVED | weak | — |
| C13 | Confidence–consistency selective gen | KILL | NAMED / CROWDED | selective generation, disagreement signals |
| C3 | Related selective/UQ backup | KILL | NAMED / CROWDED | — |
| N1 | Associated-hallucination + minimal evidence | KILL | FUTURE_WORK / FRAMEWORK | AH taxonomy; RAG verification |
| N2 | Related hallucination/evidence | KILL | CROWDED | — |
| N3 | Small-RAG conflict policy | KILL | NAMED | FRANQ and RAG-conflict line |

---

## Reboot landscape — D-series

| ID | Topic (short) | Decision | Reason |
|----|---------------|----------|--------|
| D1 | Constrained/structured decoding eval | KILL | ABLATION / FRAMEWORK (DCCD neighborhood) |
| D3 | Speculative / decoding quality tradeoff | KILL | NAMED / CROWDED |
| D4 | Synthetic curriculum / diversity predictor | KILL | CROWDED |
| D5 | Related constrained-decoding | KILL | ABLATION |
| D6 | Classical SE→LLM / Colab-as-novelty class | KILL | FRAMEWORK / EXCLUDED_NBHD |

---

## Reboot #2 — M-series (mechanism)

| ID | Topic (short) | Decision | Reason |
|----|---------------|----------|--------|
| M1 | MDM / remasking / SER neighborhood | KILL | NAMED / EXCLUDED_NBHD |
| M2 | Lazy→rich LoRA / related PEFT geometry | KILL | NAMED / COMBO |
| M4 | TinyLoRA / feature learning rate | KILL | NAMED |
| M5 | Related remasking independence | KILL | FUTURE_WORK / NAMED |

Hard exclusion after this round: entire **MDM / SER / remasking / planner–sampler** neighborhood.

---

## Reboot #3 — R3-series

| ID | Topic (short) | Decision | Reason |
|----|---------------|----------|--------|
| R3-1 … R3-5 | Interactive state / KV / equilibrium / related | KILL | NAMED / COMBO / CROWDED |

Saturated clusters noted: interactive state integrity, gen–rec asymmetry, tokenization-as-binding, multitask gradient geometry, retokenization path-patching.

---

## Reboot #4 — R4-series

| ID | Topic (short) | Decision | Reason |
|----|---------------|----------|--------|
| R4-1 … R4-4 | Follow-on mechanism dualisms | KILL | NAMED / COMBO |

---

## Reboot #5 — Morphology / phonology

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R5-1 | Frequency × complexity morphological productivity | KILL | CLEANER_SYNTH / NAMED | Petty productivity; SIGTYP 2025; Spanish morphome; PNAS analogy |
| R5-2 | Forecasted-trigger allomorphy | KILL | CLONE | arXiv:2609.04708 |
| R5-3 … R5-5 | Not warranted / not tested | ABANDONED | — | — |

---

## Reboot #6 — Unlearning / quantization

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R6-1 | Skill unlearning vs fact unlearning | KILL | NAMED | Skill Unlearning (2503.21730); Tool Unlearning; FLU; PISCES; CLUE |
| R6-2 | Computation vs representation collapse (PTQ) | KILL | CLONE / COMBO | arXiv:2604.19884 |
| R6-3 … R6-5 | Not tested | ABANDONED | — | — |

---

## Reboot #7 — Interference

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R7-1 | Dual-procedure / typed interference | KILL | NAMED | Circuit-Interference Law (2603.06923); Huang capacity–interference (2605.29548); Ortho-LoRA; LoRI; OSRM |
| R7-2 | CoT commitment as hardness law | KILL | CLONE | Decorative→load-bearing CoT (2609.25366) |

---

## Reboot #8 — Prefill / curriculum

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R8-1 | Prefill vs decode regimes | KILL | NAMED | Notes-at-prefill (2606.17107); SPEED; serving P/D literature |
| R8-2 | Training order → which circuit | KILL | NAMED | Curricula→circuits (2607.04846); Order Is The Message; multi-hop curriculum |

---

## Reboot #9 — Recency

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R9-2 | Exposure-recency vs content circuits | KILL | NAMED / CLONE | Fresh in Memory (2509.14223, ICLR 2026); PTC/TAS |
| R9-1,3,4 | Not tested after R9-2 death | ABANDONED | — | — |

---

## Reboot #10 — Attribution / consolidation

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R10-1 | Loss/attribution mediates capability consolidation | KILL | NAMED / COMBO | Capability Provenance; TokenUnlearn; IF-GUIDE; Huang; PRISM; PiKE |
| R10-2 | Seen/unseen = recency axis | SKIPPED | already claimed by Fresh in Memory | — |

---

## Reboots #11–#14 — Scout waves (immediate kills)

All scouts below died on contact with titled 2025–2026 owners (representative):

| Scout | Decision | Anchors |
|-------|----------|---------|
| Context stickiness | KILL | 2604.23371 |
| Schema ↔ Binding ICL | KILL | 2512.17325 |
| Procedural hallucination 2A/2B | KILL | 2602.19239; Legible Failures |
| Context vs memory authority | KILL | 2609.00753 |
| Variable binding / rebinding | KILL | Binding-ID line; 2505.20896; rebinding |
| Entropy / trajectory UQ | KILL | Entropy-Lens; HEAD ENTROPY; Truth as Trajectory |
| Teacher-force vs free-run | KILL | exposure bias / drift literature |
| Tokenisation fertility / bias | KILL | Causal tokenisation bias; BPE morphology papers |
| Manifold vs discrete logic | KILL | counting manifolds; geometric discrete logic |
| CLM vs MDM mechanism shift | KILL | 2601.14758 (+ MDM exclusion) |
| Gradient-Causal Gap | KILL | 2602.01442 |
| Represented ≠ computed | KILL | 2605.22488 |
| Rank collapse / Post-Norm | KILL | 2608.09417 |
| Attention sinks necessity | KILL | ACL 2026 short / 2603.11487 |
| Copy suppression | KILL | classic negative-head work |
| Self-repair / CoAx | KILL | Hydra; GIM; 2607.01940 |
| Superposition interference | KILL | 2602.04718; SAE recovery |
| Protoreasoning / Dyck CoT | KILL | 2608.04980 |

---

## Survivors

**None.**

## Hard exclusion neighborhoods (accumulated)

Do not re-propose topics inside:

1. UQ / hallucination detection / selective generation  
2. RAG-conflict policies  
3. Decoding-only / speculative tradeoffs (as novelty)  
4. Synthetic-collapse predictors (measure-only)  
5. Classical SE→LLM transfer; Colab-as-novelty  
6. MDM / SER / remasking / planner–sampler  
7. Interactive state integrity / judge-Goodhart cluster  
8. Gen–rec asymmetry; tokenization-as-binding; Ortho-LoRA / multitask geometry  
9. Morphology frequency×complexity productivity designs  
10. Phonological forecast-allomorphy  
11. Skill vs knowledge unlearning; PTQ two-failure-modes  
12. Circuit-Interference Law; capacity–interference dualisms  
13. Prefill–decode notes; curriculum→circuit; Fresh-in-Memory recency  
14. ICL schema/binding; context stickiness; authority; binding/rebinding  
15. Trajectory entropy UQ; sinks; self-repair; superposition-interference method papers  

---

## How to use this file

- Historical record for the project owner  
- Guardrail against re-proposing killed objects under new names  
- **Not** a substitute for a fresh literature review if constraints change
