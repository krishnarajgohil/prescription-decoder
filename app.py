"""
AI Medical Prescription Decoder
--------------------------------
A Streamlit app that uses Google Gemini to explain handwritten prescriptions
and lab reports in simple language for elderly and rural patients.

Run with:  streamlit run app.py
"""

import streamlit as st
import google.generativeai as genai
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# NOTE: "gemini-1.5-flash" has been retired by Google. If you get a
# "model not found" error, use a current Flash model such as the one below.
DEFAULT_MODEL = "gemini-3.8-flash"

PROMPT = """
You are a helpful, empathetic medical assistant who explains prescriptions and
lab reports to elderly or rural patients who may have little medical knowledge.
Use warm, simple, non-jargon English and short sentences.

Carefully read the attached image (it may be a messy handwritten prescription
or a lab report). Then respond using EXACTLY these sections:

## 💊 What These Medicines Are For
For each medicine you can identify, give its name and a one-or-two-sentence
plain-English explanation of what it is likely used for.
If it is a lab report, explain each key result simply and say whether it looks
normal, high, or low.

## 🗓️ Daily Schedule
Provide a clear Markdown table with these exact columns:
| Medicine Name | Morning | Afternoon | Night | Before/After Food |
Use ✅ for "take" and ❌ for "do not take". If the dose, timing, or food
instruction is not written, put "Not specified". If handwriting is unclear,
write your best guess followed by "(unclear)".

## ⚠️ Disclaimer
End with: "This is AI-generated and may contain mistakes. Always consult your
doctor or pharmacist before taking any medicine or making health decisions."

Rules:
- Never invent medicines or doses that are not visible in the image.
- If the image is not a prescription or medical report, politely say so
  instead of guessing.
"""

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Prescription Decoder", page_icon="💊", layout="centered")

st.title("💊 AI Medical Prescription Decoder")
st.write(
    "Upload a photo of a handwritten prescription or lab report, and get a "
    "simple explanation and an easy daily medicine schedule."
)

# ---------------------------------------------------------------------------
# Sidebar: API key + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Settings")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",  # masks the key on screen
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )
    model_name = st.text_input("Model name", value=DEFAULT_MODEL)
    st.caption("Your key is used only for this session and is never stored.")

    if not api_key:
        st.warning("Please enter your Gemini API key to continue.")

# ---------------------------------------------------------------------------
# Main area: file upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your prescription or lab report", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Validate that the file is a real, readable image
    try:
        image = Image.open(uploaded_file)
        image.load()  # forces decoding so corrupt files fail here
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("❌ This file could not be read as an image. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded document", use_container_width=True)

    # -----------------------------------------------------------------------
    # Decode button + Gemini call
    # -----------------------------------------------------------------------
    if st.button("Decode Prescription", type="primary"):
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key in the sidebar first.")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)

                with st.spinner("Reading your prescription... please wait"):
                    response = model.generate_content([PROMPT, image])

                # response.text raises ValueError if the reply was blocked/empty
                st.success("Here is your decoded prescription:")
                st.markdown(response.text)

            except ValueError:
                st.error(
                    "The AI returned no readable answer (the image may be unclear "
                    "or was blocked by safety filters). Try a sharper photo."
                )
            except Exception as e:  # API key errors, quota, network, bad model name...
                st.error(f"Something went wrong while contacting Gemini: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("⚕️ AI-generated information only. Not a substitute for professional medical advice.")