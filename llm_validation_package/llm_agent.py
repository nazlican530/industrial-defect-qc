import json
import os
from typing import Any

from groq import Groq


def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not defined."
        )
    return Groq(api_key=api_key)


def generate_report(
    defect_class: str,
    confidence: float,
    severity: str,
    decision: str,
) -> dict[str, Any]:
    """
    Generate a structured industrial inspection report.

    The function returns only the LLM output. Validation and fallback handling
    are performed by the backend or the standalone evaluation script.
    """
    system_prompt = """
You are an industrial quality-control reporting assistant.

Return ONLY one valid JSON object.
Do not use Markdown.
Do not add text outside the JSON object.
Use exactly these keys:
defect_description
risk_level
recommended_action
production_impact
confidence_interpretation

Requirements:
- defect_description must explicitly mention the supplied defect class.
- risk_level must match the supplied severity exactly.
- recommended_action must match the supplied decision exactly.
- confidence_interpretation must be consistent with the supplied confidence.
- Keep all values short, professional, and factual.
- Do not invent measurements, causes, standards, or production details.
""".strip()

    user_prompt = f"""
Defect Class: {defect_class}
Confidence: {confidence:.4f}
Severity: {severity}
Decision: {decision}
""".strip()

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Groq API request failed: {exc}"
        ) from exc

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("The LLM returned an empty response.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"The LLM returned invalid JSON: {exc}"
        ) from exc

    required_fields = [
        "defect_description",
        "risk_level",
        "recommended_action",
        "production_impact",
        "confidence_interpretation",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in data or not str(data[field]).strip()
    ]

    if missing_fields:
        raise RuntimeError(
            "The LLM response is missing required fields: "
            + ", ".join(missing_fields)
        )

    return {
        "defect_description": str(data["defect_description"]).strip(),
        "risk_level": str(data["risk_level"]).strip(),
        "recommended_action": str(data["recommended_action"]).strip(),
        "production_impact": str(data["production_impact"]).strip(),
        "confidence_interpretation": str(
            data["confidence_interpretation"]
        ).strip(),
    }
