import os
from google import genai

def generate_adaptive_response(query,persona,chunks):
    ctx="\n\n".join([c["text"] for c in chunks])
    styles={
      "Technical Expert":"Give detailed technical steps and diagnostics.",
      "Frustrated User":"Be empathetic and provide simple numbered steps.",
      "Business Executive":"Be concise and focus on impact and timeline."
    }
    client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt=f'''{styles.get(persona,"")}
Use only this context:
{ctx}
Question:{query}'''
    r=client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
    return r.text
