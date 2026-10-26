import streamlit as st
import google.generativeai as genai

API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

def generate_answer(
    question,
    retrieved_context
):

    prompt = f"""
You are an AI Surveillance Assistant.

Question:
{question}

Retrieved Records:
{retrieved_context}

Instructions:

- Use only the events explicitly present in Retrieved Records.
- Never infer, reinterpret, or substitute event names.
- "Fall Detected", "High Risk", "Running", "Restricted Area", and "Global Loitering" are different events.
- If the requested event is not present in Retrieved Records, reply:
  "No incidents found for the requested event."
- If no snapshot exists for the requested event, reply:
  "No images available for the requested event."
- Keep answers concise and factual.
"""

    response = model.generate_content(
        prompt
    )

    return response.text