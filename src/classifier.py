# import os,json
# from google import genai

# def classify_customer_persona(user_message:str)->dict:
#     client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
#     prompt=f'''Classify as one of: Technical Expert, Frustrated User, Business Executive.
# Return JSON with persona, confidence, reasoning.
# Message: {user_message}'''
#     r=client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
#     txt=r.text.strip()
#     try:return json.loads(txt)
#     except:return {"persona":"Technical Expert","confidence":0.5,"reasoning":"fallback"}

import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

def classify_customer_persona(message):

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    prompt = f"""
Classify this support message into exactly one category:

1. Technical Expert
2. Frustrated User
3. Business Executive

Return ONLY valid JSON.

Example:
{{
    "persona":"Technical Expert",
    "confidence":0.95,
    "reasoning":"Uses technical terminology."
}}

Message:
{message}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        text = response.text.strip()
        text = text.replace("```json", "")
        text = text.replace("```", "")

        return json.loads(text)

    except Exception as e:
        print(e)

        return {
            "persona": "Technical Expert",
            "confidence": 0.5,
            "reasoning": "fallback"
        }