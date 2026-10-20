import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.quality_gate import evaluate_quality_gate


def test_quality_gate_passes_when_all_thresholds_are_met():
    metrics = {
        "accuracy": 0.84,
        "precision": 0.55,
        "recall": 0.90,
        "f1": 0.68,
        "roc_auc": 0.92,
    }
    thresholds = {
        "accuracy": 0.80,
        "precision": 0.50,
        "recall": 0.80,
        "f1": 0.60,
        "roc_auc": 0.85,
    }

    results = evaluate_quality_gate(metrics, thresholds)
    assert all(item["passed"] for item in results.values())


def test_quality_gate_fails_when_a_threshold_is_missed():
    metrics = {
        "accuracy": 0.84,
        "precision": 0.40,
        "recall": 0.90,
        "f1": 0.68,
        "roc_auc": 0.92,
    }
    thresholds = {
        "accuracy": 0.80,
        "precision": 0.50,
        "recall": 0.80,
        "f1": 0.60,
        "roc_auc": 0.85,
    }

    results = evaluate_quality_gate(metrics, thresholds)
    assert not results["precision"]["passed"]
    assert not all(item["passed"] for item in results.values())


def test_quality_gate_rejects_low_recall():
    metrics = {
        "accuracy": 0.84,
        "precision": 0.55,
        "recall": 0.60,
        "f1": 0.68,
        "roc_auc": 0.92,
    }
    thresholds = {
        "accuracy": 0.80,
        "precision": 0.50,
        "recall": 0.80,
        "f1": 0.60,
        "roc_auc": 0.85,
    }

    results = evaluate_quality_gate(metrics, thresholds)
    assert results["recall"]["passed"] is False
    assert not all(item["passed"] for item in results.values())


def test_quality_gate_rejects_missing_required_metric():
    metrics = {
        "accuracy": 0.84,
        "precision": 0.55,
        "recall": 0.90,
        "f1": 0.68,
    }
    thresholds = {
        "accuracy": 0.80,
        "precision": 0.50,
        "recall": 0.80,
        "f1": 0.60,
        "roc_auc": 0.85,
    }

    results = evaluate_quality_gate(metrics, thresholds)
    assert not all(item["passed"] for item in results.values())
