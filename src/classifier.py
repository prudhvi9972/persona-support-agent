import os,json
from google import genai

def classify_customer_persona(user_message:str)->dict:
    client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt=f'''Classify as one of: Technical Expert, Frustrated User, Business Executive.
Return JSON with persona, confidence, reasoning.
Message: {user_message}'''
    r=client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
    txt=r.text.strip()
    try:return json.loads(txt)
    except:return {"persona":"Technical Expert","confidence":0.5,"reasoning":"fallback"}
