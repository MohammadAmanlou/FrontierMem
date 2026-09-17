# Tomorrow: Zahra Meeting Checklist

## Show first (2 minutes)

1. Existing repository already contains:
   - 4,608-example legacy controlled diagnostic.
   - rule/classical/tiny-model baselines.
   - Qwen2.5 0.5B/1.5B/3B experiments.
   - evidence that memory extraction and final applicability decisions fail
     differently.

2. New scientific direction:
   - **Preference Decision Identifiability**
   - We do not need to identify the whole user state.
   - We need to know whether all plausible evidence supports the same
     personalization decision.
   - If not, clarify.

3. Important change from the previous proposal:
   - no new synthetic benchmark as the paper's center.
   - released datasets become the primary empirical substrate.

## Concrete data plan to agree on

### Train / development
- RPEval `data_generation` pool.
- Later, exact task-specific S2Pref training records if licensing/task protocol
  permits and after schema verification.

### Held-out direct applicability tests
- RPEval explicit and implicit benchmark test sets.
- BenchPreS, external and strictly evaluation-only.

### Transfer tests
- S2Pref clarification/context tasks.
- RealPref unseen-scenario/long-horizon tests.
- PersonaMem-v2 long-context personalization.
- AlpsBench real-dialogue Task4 preference/constraint tracks.

## Code to show

- `frontiermem/external_data.py`
  - source-preserving adapters
  - no generated language
  - explicit label mappings

- `frontiermem/contrast_sets.py`
  - creates same-preference APPLY/IGNORE pairs from existing records only

- `frontiermem/identifiability.py`
  - executable decision-identifiability rule
  - conformal abstention -> CLARIFY

- `scripts/train_caid_pairwise.py`
  - pointwise applicability loss + pairwise margin loss
  - LoRA-capable 3B/7B model training

## Ask Zahra to decide four things

1. Do we freeze **Decision Identifiability** as the central theoretical object?
2. Is RPEval + BenchPreS the right first direct-applicability stack?
3. For S2Pref, should we use its published clarification task as the primary
   CLARIFY evaluation rather than inventing our own ambiguous examples?
4. Should the first real-model comparison be exactly:
   - pointwise pretrained classifier
   - pairwise CAID
   - pairwise CAID + calibrated abstention?

## What not to spend meeting time on

- Neo4j / memory database engineering.
- RL/GRPO.
- expanding the synthetic generator.
- additional Qwen prompt-only scaling.
- cosmetic architecture changes.

## Desired meeting outcome

Leave with:
- frozen problem statement,
- frozen dataset stack,
- frozen first 3 experiments,
- exact first GPU model size,
- decision on the S2Pref clarification protocol.
