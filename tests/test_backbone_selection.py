import os

import pytest

from frontiermem.dual_axis_protocol import summary
from scripts.hpc_model_env_preflight import MODELS, inspect
from scripts.train_dual_axis_qwen import parse_args


def test_new_backbones_pinned_and_not_historical():
    assert MODELS["primary_ablation"] == "Qwen/Qwen3.5-9B"
    assert MODELS["flagship_large"] == "Qwen/Qwen3.8-27B"
    assert MODELS["historical_only"] != MODELS["primary_ablation"]


def test_default_arm_model_matches_reported_id():
    args = parse_args(["--arm", "dual_axis", "--dry-run"])
    assert args.model == MODELS["primary_ablation"]
    fake = {"rpeval_train": [], "rpeval_dev": [],
            "s2pref_train": [], "s2pref_dev": [],
            "s2pref_strict_dev": []}
    assert summary(fake, "dual_axis", args.model)["model"] == args.model


def test_flagship_dry_run_keeps_selected_model():
    args = parse_args(["--arm", "dual_axis", "--dry-run",
                       "--model", MODELS["flagship_large"]])
    assert args.model == MODELS["flagship_large"]


def test_login_env_check_does_not_load_model():
    report = inspect(False)
    assert report["status"] == "READ_ONLY_NO_MODEL_DOWNLOAD"
    assert "installed_packages" in report
    assert "models_to_evaluate" in report
    assert "pytorch" not in report


def test_gpu_env_check_rejects_login_node(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        inspect(True)
