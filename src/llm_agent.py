import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_report(defect_class, confidence, severity, decision):

    system_prompt = """
You are an industrial quality control AI assistant.

You must return ONLY valid JSON.
Do NOT return markdown.
Do NOT explain anything outside JSON.
Do NOT override the given severity or decision.
"""

    user_prompt = f"""
Defect Class: {defect_class}
Confidence: {confidence:.2f}
Severity: {severity}
Decision: {decision}

Return JSON in this exact format:

{{
  "defect_description": "...",
  "risk_level": "...",
  "recommended_action": "...",
  "production_impact": "...",
  "confidence_interpretation": "..."
}}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except:
        return {
            "defect_description": "LLM parsing failed",
            "risk_level": "Unknown",
            "recommended_action": "Manual review required",
            "production_impact": "Unknown",
            "confidence_interpretation": "Parsing error"
        }