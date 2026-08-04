import csv
import json
import statistics
import time
from collections import Counter
from itertools import cycle
from pathlib import Path
from typing import Any

from src.llm_agent import generate_report
from src.llm_validation_utils import validate_llm_report


OUTPUT_DIR = Path("outputs/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFECT_CASES = [
    ("scratches", "Low"),
    ("crazing", "Medium"),
    ("inclusion", "Medium"),
    ("pitted_surface", "Medium"),
    ("patches", "High"),
    ("rolled-in-scale", "High"),
]

CONFIDENCE_CASES = [
    0.35,
    0.50,
    0.65,
    0.80,
    0.92,
    0.97,
]


def severity_based_decision(severity: str) -> str:
    if severity == "Low":
        return "ACCEPT"
    if severity == "Medium":
        return "REWORK"
    if severity == "High":
        return "REJECT"
    raise ValueError(f"Unknown severity: {severity}")


def operational_decision(
    confidence: float,
    severity: str,
    top2_gap: float,
) -> str:
    """
    Decision logic aligned with the manuscript:

    - Top-2 gap < 0.10 -> HUMAN REVIEW
    - Confidence < 0.90 -> HUMAN REVIEW
    - Confidence >= 0.90 -> severity-based decision
    """
    if top2_gap < 0.10:
        return "HUMAN REVIEW"

    if confidence < 0.90:
        return "HUMAN REVIEW"

    return severity_based_decision(severity)


def build_cases(total: int = 100) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    defect_cycle = cycle(DEFECT_CASES)
    confidence_cycle = cycle(CONFIDENCE_CASES)

    for case_id in range(1, total + 1):
        defect_class, severity = next(defect_cycle)
        confidence = next(confidence_cycle)

        # Every fifth case is made ambiguous to test the Top-2 rule.
        top2_gap = 0.05 if case_id % 5 == 0 else 0.20

        decision = operational_decision(
            confidence=confidence,
            severity=severity,
            top2_gap=top2_gap,
        )

        cases.append({
            "case_id": case_id,
            "defect_class": defect_class,
            "confidence": confidence,
            "severity": severity,
            "top2_gap": top2_gap,
            "decision": decision,
        })

    return cases


def run_evaluation(total: int = 100) -> None:
    rows: list[dict[str, Any]] = []

    for case in build_cases(total):
        start_time = time.perf_counter()

        status = "ok"
        report: dict[str, Any] | None = None
        errors: list[str] = []

        try:
            report = generate_report(
                defect_class=case["defect_class"],
                confidence=case["confidence"],
                severity=case["severity"],
                decision=case["decision"],
            )

            errors = validate_llm_report(
                report=report,
                defect_class=case["defect_class"],
                confidence=case["confidence"],
                severity=case["severity"],
                decision=case["decision"],
            )

            if errors:
                status = "fallback_used"

        except Exception as exc:
            status = "llm_failed"
            errors = [str(exc)]

        latency_seconds = time.perf_counter() - start_time

        row = {
            **case,
            "status": status,
            "validation_passed": status == "ok",
            "errors": " | ".join(errors),
            "latency_seconds": round(latency_seconds, 4),
            "report_json": json.dumps(
                report,
                ensure_ascii=False,
            ) if report is not None else "",
        }
        rows.append(row)

        print(
            f"{case['case_id']:03d}/{total} | "
            f"{status:<13} | "
            f"{errors if errors else 'valid'}"
        )

    csv_path = OUTPUT_DIR / "llm_validation_results.csv"
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(row["status"] for row in rows)
    error_counts = Counter()

    for row in rows:
        if row["errors"]:
            for error in row["errors"].split(" | "):
                error_counts[error] += 1

    latencies = [
        float(row["latency_seconds"])
        for row in rows
    ]

    total_reports = len(rows)
    validated_reports = status_counts["ok"]
    fallback_reports = status_counts["fallback_used"]
    llm_failures = status_counts["llm_failed"]

    summary = {
        "total_reports": total_reports,
        "validated_reports": validated_reports,
        "fallback_reports": fallback_reports,
        "llm_failures": llm_failures,
        "validation_pass_rate_percent": round(
            validated_reports / total_reports * 100,
            2,
        ),
        "fallback_rate_percent": round(
            fallback_reports / total_reports * 100,
            2,
        ),
        "llm_failure_rate_percent": round(
            llm_failures / total_reports * 100,
            2,
        ),
        "mean_latency_seconds": round(
            statistics.mean(latencies),
            4,
        ),
        "median_latency_seconds": round(
            statistics.median(latencies),
            4,
        ),
        "latency_std_seconds": round(
            statistics.pstdev(latencies),
            4,
        ),
        "validation_error_counts": dict(error_counts),
    }

    summary_path = OUTPUT_DIR / "llm_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    text_path = OUTPUT_DIR / "llm_validation_summary.txt"
    text_path.write_text(
        "\n".join([
            "===== LLM VALIDATION SUMMARY =====",
            f"Total reports:          {total_reports}",
            f"Validated reports:      {validated_reports}",
            f"Fallback reports:       {fallback_reports}",
            f"LLM failures:           {llm_failures}",
            (
                "Validation pass rate:   "
                f"{summary['validation_pass_rate_percent']:.2f}%"
            ),
            (
                "Fallback rate:          "
                f"{summary['fallback_rate_percent']:.2f}%"
            ),
            (
                "LLM failure rate:       "
                f"{summary['llm_failure_rate_percent']:.2f}%"
            ),
            (
                "Mean latency:           "
                f"{summary['mean_latency_seconds']:.4f} s"
            ),
            (
                "Median latency:         "
                f"{summary['median_latency_seconds']:.4f} s"
            ),
            (
                "Latency std.:           "
                f"{summary['latency_std_seconds']:.4f} s"
            ),
            (
                "Validation errors:      "
                f"{dict(error_counts)}"
            ),
        ]),
        encoding="utf-8",
    )

    print("\n===== LLM VALIDATION SUMMARY =====")
    print(f"Total reports:        {total_reports}")
    print(f"Validated reports:    {validated_reports}")
    print(f"Fallback reports:     {fallback_reports}")
    print(f"LLM failures:         {llm_failures}")
    print(
        "Validation pass rate: "
        f"{summary['validation_pass_rate_percent']:.2f}%"
    )
    print(
        "Fallback rate:        "
        f"{summary['fallback_rate_percent']:.2f}%"
    )
    print(
        "Mean latency:         "
        f"{summary['mean_latency_seconds']:.4f} s"
    )
    print(f"\nCSV saved to: {csv_path}")
    print(f"JSON saved to: {summary_path}")
    print(f"TXT saved to: {text_path}")


if __name__ == "__main__":
    run_evaluation(total=100)
