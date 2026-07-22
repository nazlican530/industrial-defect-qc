import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def generate_report(defect_class, confidence, severity, decision):
    system_prompt = """
You are an industrial quality control AI assistant.

Return ONLY valid JSON.
Do not use markdown.
Do not add explanations outside JSON.
Keep the values short and professional.
Use exactly these keys:
defect_description,
risk_level,
recommended_action,
production_impact,
confidence_interpretation
"""

    user_prompt = f"""
Defect Class: {defect_class}
Confidence: {confidence:.2f}
Severity: {severity}
Decision: {decision}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    print("LLM RESPONSE:", content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Invalid JSON returned by LLM: {e}"
        ) from e

    required_fields = [
        "defect_description",
        "risk_level",
        "recommended_action",
        "production_impact",
        "confidence_interpretation",
    ]

    missing = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing:
        raise RuntimeError(
            f"Missing required fields: {', '.join(missing)}"
        )

    return {
        "defect_description": data["defect_description"],
        "risk_level": data["risk_level"],
        "recommended_action": data["recommended_action"],
        "production_impact": data["production_impact"],
        "confidence_interpretation": data["confidence_interpretation"],
    }