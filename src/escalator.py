from src.config import CONFIDENCE_THRESHOLD,SENSITIVE_KEYWORDS
import json

def check_escalation(query,chunks):
    score=max([c["score"] for c in chunks],default=0)
    if score<CONFIDENCE_THRESHOLD or any(k in query.lower() for k in SENSITIVE_KEYWORDS):
        return True,json.dumps({"issue":query,"confidence":score,"action":"Human review required"},indent=2)
    return False,None
