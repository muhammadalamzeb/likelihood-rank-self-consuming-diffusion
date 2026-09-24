# RESEARCH LOG

## 2026-09-23 — Ownership mandate start

- **Hypothesis:** N/A (process)
- **Evidence:** Path 1–3 terminal; Path 2 v3 NO KEEP; Path 4 archive; owner grants full pipeline + Constraint Set v4
- **Literature:** Status docs read; env verified (Python 3.14, torch 2.14 CPU)
- **Decision:** **KEEP** project open under v4; begin Wave-2 discovery
- **Reason:** Explicit ownership + bar change authorizes resume past Path 4
- **Experiment ID:** —
- **Conclusion:** Research reopened

## 2026-09-23 — Wave-2 discovery + kill-tests

- **Hypothesis:** Multiple (W2-1…W2-10); see `PATH2_DISCOVERY.md`
- **Evidence:** arXiv/HTML reads of collapse, verification, LSF, temperature, TJS, exposure-bias, long-tail diffusion, recon-vs-gen
- **Literature checked:** 2305.17493; 2404.01413; 2307.01850; 2406.07515; 2511.12742; 2607.10853; 2510.01184; 2607.06114; CVPR2025 Yao recon-vs-gen; exposure-bias line; Path-2 priors
- **Decision:** **KILL** W2-2…W2-9; **KEEP** provisional **W2-1**
- **Reason:** Only W2-1 clears “what is new after closest papers?” as false-stability under likelihood top‑k vs random‑k
- **Experiment ID:** pending `w2_fs`
- **Conclusion:** Documented in `RESEARCH_SURVIVOR.md`; proceed to implement

## 2026-09-23 — Implementation start

- **Hypothesis:** W2-1 false stability
- **Decision:** **KEEP** → implement
- **Experiment ID:** `w2_fs`
- **Conclusion:** Coding + runs next

## 2026-09-23 — Pilot quick run → MODIFY

- **Hypothesis:** W2-1 (W₂ better + minority worse under top‑k)
- **Evidence:** `w2_fs.jsonl` quick: top‑k minority_mean < rand‑k across seeds, but sliced W₂ for top‑k ≥ rand‑k (no aggregate improvement)
- **Literature:** unchanged
- **Decision:** **MODIFY** claim to majority-concentration / minority-extinction matched‑k effect; drop “better W₂” half
- **Reason:** Automatic pivot rule — do not silently keep falsified dissociation
- **Experiment ID:** `w2_fs`
- **Conclusion:** Revise survivor; run full matrix

## 2026-09-24 — Full GMM matrix + Digits transfer

- **Hypothesis:** Revised W2-1 — likelihood-rank ordering of minority retention
- **Evidence:** Retention g1/g0 means bottom 0.457 / rand 0.289 / top 0.161; ρ=5 g=1 all seeds bottom>rand>top; W₂ false stability falsified; Digits entropy weakly ordered but class-mass unreliable
- **Literature:** MAD, Feng, LSF (as in survivor)
- **Decision:** **KEEP** revised claim for workshop-scope paper; report Digits as inconclusive
- **Experiment ID:** `w2_fs`, `w2_fs_digits`
- **Conclusion:** Write `paper/PAPER.md` + repro package; PUBLICATION-READY (L2 empirical scope)

## 2026-09-24 — Replication seeds 3–4 + Wilcoxon stats

- **Hypothesis:** Same revised W2-1
- **Evidence:** Merged log n=228; retention bottom/rand/top = 0.572/0.360/0.172 (n=8 pairs); order holds 5/5 seeds at ρ=5 g=1; Wilcoxon p≈0.042 for top−rand and bottom−rand retention
- **Decision:** **KEEP**; update paper numbers + `scripts/reproduce_w2.ps1`; add `--append` to runner
- **Experiment ID:** `w2_fs`
- **Conclusion:** Stronger replication package committed

## 2026-09-24 — Alpha ablation + LaTeX package

- **Hypothesis:** Ordering robust across real-mix α
- **Evidence:** `w2_fs_alpha.jsonl`: at α=0.25/0.5 top worst; at α=0.75 top−rand gap reverses while bottom still highest mean retention
- **Decision:** **KEEP** with scoped α limitation; add `paper/main.tex`, `refs.bib`
- **Experiment ID:** `w2_fs_alpha`
- **Conclusion:** Submission package extended

## 2026-09-24 — 8D transfer + k_frac ablation

- **Hypothesis:** Ordering transfers beyond 2D; sensitive to selection pool size
- **Evidence:** HD `w2_fs_hd`: 3/3 seeds order holds, retention 0.55/0.27/0.08; k_frac 0.25≡0.5 under n_syn binding; at 0.75 top still worst
- **Decision:** **KEEP**; update paper with transfer + pool-size notes
- **Experiment ID:** `w2_fs_hd`, `w2_fs_kfrac`
- **Conclusion:** Stronger robustness section
