import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
 # apiden alınan anahtarı kullanarak Groq istemcisini başlatıyoruz. Bu istemci, LLM ile iletişim kurmak için kullanılacak.
def generate_report(defect_class, confidence, severity, decision):
    system_prompt = """
You are an industrial quality control AI assistant.

Return ONLY valid JSON.
Do not use markdown.
Do not add explanations outside JSON.
Keep the values short and professional.
Use exactly these keys:
defect_description, risk_level, recommended_action, production_impact, confidence_interpretation
"""

    user_prompt = f"""
Defect Class: {defect_class}
Confidence: {confidence:.2f}
Severity: {severity}
Decision: {decision}
"""

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

        content = response.choices[0].message.content
        print("LLM RESPONSE:", content)

        data = json.loads(content)

        return {
            "defect_description": data.get("defect_description", defect_class),
            "risk_level": data.get("risk_level", severity),
            "recommended_action": data.get("recommended_action", decision),
            "production_impact": data.get("production_impact", "Unknown"),
            "confidence_interpretation": data.get("confidence_interpretation", "Unknown"),
        }

    except Exception as e:
        print("LLM ERROR:", str(e))
        return {
            "defect_description": "LLM parsing failed",
            "risk_level": "Unknown",
            "recommended_action": "Manual review required", # kararın ne olduğunu bilmediğimiz için manuel inceleme öneriyoruz.
            "production_impact": "Unknown",
            "confidence_interpretation": "Parsing error"
        }