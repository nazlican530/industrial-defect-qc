import re
from typing import Any


REQUIRED_REPORT_FIELDS = {
    "defect_description",
    "risk_level",
    "recommended_action",
    "production_impact",
    "confidence_interpretation",
}


def normalize_text(value: Any) -> str:
    text = str(value).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def expected_confidence_label(confidence: float) -> str:
    if confidence >= 0.90:
        return "very confident"
    if confidence >= 0.75:
        return "high confidence"
    if confidence >= 0.60:
        return "moderate confidence"
    if confidence >= 0.40:
        return "low confidence"
    return "very low confidence"


def validate_llm_report(
    report: dict[str, Any],
    defect_class: str,
    confidence: float,
    severity: str,
    decision: str,
) -> list[str]:
    """
    Validate an LLM report against structured model outputs.

    The function checks:
    1. report type and required fields,
    2. predicted defect consistency,
    3. severity consistency,
    4. decision consistency,
    5. confidence-interpretation consistency,
    6. unsupported high-risk language for low-severity cases.
    """
    errors: list[str] = []

    if not isinstance(report, dict):
        return ["Invalid report type"]

    missing_fields = sorted(
        field
        for field in REQUIRED_REPORT_FIELDS
        if field not in report or not str(report[field]).strip()
    )

    if missing_fields:
        errors.append(
            "Missing fields: " + ", ".join(missing_fields)
        )

    expected_defect = normalize_text(defect_class)
    defect_description = normalize_text(
        report.get("defect_description", "")
    )
    if expected_defect not in defect_description:
        errors.append("Defect-class mismatch")

    expected_severity = normalize_text(severity)
    actual_severity = normalize_text(report.get("risk_level", ""))
    if actual_severity != expected_severity:
        errors.append("Severity mismatch")

    expected_decision = normalize_text(decision)
    actual_decision = normalize_text(
        report.get("recommended_action", "")
    )
    if expected_decision not in actual_decision:
        errors.append("Decision mismatch")

    expected_confidence = expected_confidence_label(confidence)
    actual_confidence = normalize_text(
        report.get("confidence_interpretation", "")
    )
    if expected_confidence not in actual_confidence:
        errors.append("Confidence-interpretation mismatch")

    full_text = normalize_text(" ".join(
        str(value) for value in report.values()
    ))

    unsupported_phrases = (
        "critical risk",
        "catastrophic",
        "severe defect",
        "immediate shutdown",
    )

    if severity.lower() == "low" and any(
        phrase in full_text for phrase in unsupported_phrases
    ):
        errors.append("Unsupported high-risk language")

    return errors
