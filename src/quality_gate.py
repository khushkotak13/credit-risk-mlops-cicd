
import json
import sys
from pathlib import Path


def evaluate_quality_gate(metrics, thresholds):
    """Return pass/fail results for configured minimum metrics."""
    results = {}

    for metric, minimum in thresholds.items():
        actual = metrics.get(metric)

        passed = actual is not None and actual >= minimum

        results[metric] = {
            "actual": actual,
            "minimum": minimum,
            "passed": passed,
        }

    return results


def main():
    metrics_path = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/metrics.json")
    thresholds_path = Path("configs/ci.yaml")

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}"
        )

    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"CI configuration not found: {thresholds_path}"
        )

    import yaml

    metrics = json.loads(metrics_path.read_text())
    config = yaml.safe_load(thresholds_path.read_text())

    thresholds = config.get("quality_gate", {}).get("thresholds", {})

    if not thresholds:
        raise ValueError(
            "No quality_gate.thresholds found in configs/ci.yaml"
        )

    results = evaluate_quality_gate(metrics, thresholds)

    print("QUALITY GATE RESULTS")
    print(json.dumps(results, indent=2))

    passed = all(result["passed"] for result in results.values())
    print("\\nQUALITY GATE:", "PASSED" if passed else "FAILED")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
