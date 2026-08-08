# Repository guide

- `frontiermem/`: core dataset, memory, baselines, tiny model, evaluation, and Hugging Face backend.
- `run_qwen_fast_ablation.py`: current Qwen ablation runner used for 1.5B/3B experiments.
- `scripts/compare_qwen_models.py`: reusable aggregation/comparison script.
- `notebooks/FrontierMem_Kaggle_Experiment_Log.ipynb`: exact latest uploaded Kaggle notebook with execution outputs.
- `notebooks/FrontierMem_Kaggle_Clean_Experiments.ipynb`: cleaned experiment notebook with duplicated/debug cells removed and export code added.
- `results/`: original local/0.5B results plus organized `qwen_1_5b`, `qwen_3b`, and `comparisons` folders.
- `reports/concept/`: FrontierMem research concept report (PDF + LaTeX).
- `reports/qwen_1_5b/`: compact experimental report, LaTeX source, figures, and report CSVs.
- `reports/archive/`: earlier project-idea documents retained for provenance.

For a new Qwen run:
```bash
python run_qwen_fast_ablation.py --model Qwen/Qwen2.5-3B-Instruct --mode gold_memory_llm --max-examples 72
```
Then aggregate everything with:
```bash
python scripts/compare_qwen_models.py --results results --output results/comparisons
```
