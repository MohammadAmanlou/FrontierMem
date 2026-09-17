# Apply this patch to `MohammadAmanlou/FrontierMem`

This package contains **additive files only**. It does not overwrite the current
legacy generator, previous results, or Qwen reports.

From the root of your local FrontierMem clone:

```bash
# Unzip the package somewhere, then copy its contents over the repo root.
cp -r /path/to/FrontierMem_existing_data_v2_patch/* .

# Optional: append the new README section manually.
cat README_V2_SECTION.md >> README.md

git status
pytest -q tests/test_identifiability.py tests/test_external_data.py
git add frontiermem scripts tests docs requirements-external.txt \
        requirements-train.txt run_external_pipeline.sh \
        README_V2_SECTION.md APPLY_PATCH_INSTRUCTIONS.md
git commit -m "Add existing-data CAID pipeline and external benchmark adapters"
git push
```

Then run:

```bash
bash run_external_pipeline.sh
```

If `RPEval/data_generation/data.json` is still a Git-LFS pointer or unavailable,
download/clone the upstream RPEval repository with Git LFS and rerun:

```bash
git lfs install
git clone https://github.com/XueyangFeng/RPEval data/external_raw/RPEval
git -C data/external_raw/RPEval lfs pull

python scripts/prepare_external_benchmarks.py \
  --rpeval-dir data/external_raw/RPEval \
  --benchpres \
  --output-dir data/external_processed
```

Do **not** use `--allow-eval-training` for reported results.
