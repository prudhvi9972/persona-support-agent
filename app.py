import streamlit as st
from dotenv import load_dotenv
from src.classifier import classify_customer_persona
from src.rag_pipeline import LocalRAGPipeline
from src.generator import generate_adaptive_response
from src.escalator import check_escalation

load_dotenv()
st.title("Persona Adaptive Support Agent")

if "rag" not in st.session_state:
    st.session_state.rag=LocalRAGPipeline()

q=st.text_area("Customer Message")

if st.button("Submit"):
    persona=classify_customer_persona(q)
    chunks=st.session_state.rag.retrieve(q,3)
    esc,handoff=check_escalation(q,chunks)

    st.write("Persona:",persona)

    if esc:
        st.error("Escalated to Human Agent")
        st.code(handoff,language="json")
    else:
        st.success(generate_adaptive_response(q,persona["persona"],chunks))
