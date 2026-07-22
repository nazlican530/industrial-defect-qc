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


def expected_confidence_category(confidence: float) -> str:
    if confidence >= 0.90:
        return "very_high"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.60:
        return "moderate"
    if confidence >= 0.40:
        return "low"
    return "very_low"


def detect_confidence_category(text: str) -> str | None:
    normalized = normalize_text(text)

    very_high_phrases = (
        "very high",
        "very high confidence",
        "very confident",
        "extremely confident",
        "extremely high confidence",
        "highly confident",
        "strong confidence",
        "strongly confident",
        "high reliability",
        "highly reliable",
        "very reliable",
    )

    high_phrases = (
        "high",
        "high confidence",
        "confident",
        "confident prediction",
        "good confidence",
        "reasonably confident",
        "reliable",
        "reliable prediction",
    )

    moderate_phrases = (
        "moderate",
        "moderate confidence",
        "medium",
        "medium confidence",
        "moderately confident",
        "reasonable confidence",
        "some uncertainty",
    )

    low_phrases = (
        "low",
        "low confidence",
        "limited confidence",
        "uncertain",
        "uncertain prediction",
        "not fully reliable",
    )

    very_low_phrases = (
        "very low",
        "very low confidence",
        "extremely low confidence",
        "highly uncertain",
        "very uncertain",
        "unreliable",
        "unreliable prediction",
    )

    # Check more specific phrases first.
    if any(phrase == normalized or phrase in normalized for phrase in very_low_phrases):
        return "very_low"

    if any(phrase == normalized or phrase in normalized for phrase in very_high_phrases):
        return "very_high"

    if any(phrase == normalized or phrase in normalized for phrase in moderate_phrases):
        return "moderate"

    if any(phrase == normalized or phrase in normalized for phrase in low_phrases):
        return "low"

    if any(phrase == normalized or phrase in normalized for phrase in high_phrases):
        return "high"

    return None


def extract_numeric_confidence(text: str) -> float | None:
    normalized = normalize_text(text)

    percentage_matches = re.findall(
        r"\b(\d{1,3}(?:\.\d+)?)\s*%",
        normalized,
    )

    for match in percentage_matches:
        value = float(match)
        if 0 <= value <= 100:
            return value / 100.0

    decimal_matches = re.findall(
        r"\b0(?:\.\d+)?\b|\b1(?:\.0+)?\b",
        normalized,
    )

    for match in decimal_matches:
        value = float(match)
        if 0 <= value <= 1:
            return value

    return None


def confidence_categories_are_compatible(
    expected_category: str,
    detected_category: str,
) -> bool:
    compatible_categories = {
        "very_high": {"very_high", "high"},
        "high": {"very_high", "high", "moderate"},
        "moderate": {"high", "moderate", "low"},
        "low": {"moderate", "low", "very_low"},
        "very_low": {"low", "very_low"},
    }

    return detected_category in compatible_categories.get(
        expected_category,
        set(),
    )


def validate_confidence_interpretation(
    confidence_text: str,
    confidence: float,
) -> bool:
    normalized = normalize_text(confidence_text)

    if not normalized:
        return False

    expected_category = expected_confidence_category(confidence)

    numeric_confidence = extract_numeric_confidence(normalized)

    if numeric_confidence is not None:
        numeric_category = expected_confidence_category(numeric_confidence)

        if confidence_categories_are_compatible(
            expected_category,
            numeric_category,
        ):
            return True

    detected_category = detect_confidence_category(normalized)

    if detected_category is None:
        return False

    return confidence_categories_are_compatible(
        expected_category,
        detected_category,
    )


def validate_llm_report(
    report: dict[str, Any],
    defect_class: str,
    confidence: float,
    severity: str,
    decision: str,
) -> list[str]:
    """
    Validate an LLM report against structured model outputs.

    Checks:
    1. Report type and required fields
    2. Predicted defect consistency
    3. Severity consistency
    4. Decision consistency
    5. Confidence-interpretation consistency
    6. Unsupported high-risk language for low-severity cases
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
    actual_severity = normalize_text(
        report.get("risk_level", "")
    )

    if actual_severity != expected_severity:
        errors.append("Severity mismatch")

    expected_decision = normalize_text(decision)
    actual_decision = normalize_text(
        report.get("recommended_action", "")
    )

    if expected_decision not in actual_decision:
        errors.append("Decision mismatch")

    confidence_interpretation = str(
        report.get("confidence_interpretation", "")
    )

    if not validate_confidence_interpretation(
        confidence_interpretation,
        confidence,
    ):
        errors.append("Confidence-interpretation mismatch")

    full_text = normalize_text(
        " ".join(
            str(value)
            for value in report.values()
        )
    )

    unsupported_phrases = (
        "critical risk",
        "catastrophic",
        "severe defect",
        "immediate shutdown",
    )

    if normalize_text(severity) == "low" and any(
        phrase in full_text
        for phrase in unsupported_phrases
    ):
        errors.append("Unsupported high-risk language")

    return errors
