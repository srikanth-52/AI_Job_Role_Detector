import streamlit as st
from utils.predictor import predict_job
import PyPDF2

st.set_page_config(page_title="Job Detector", layout="centered")

st.title("💼 AI Job Role Detector")
st.markdown("Enter skills or upload resume 🚀")

# ----------- TEXT INPUT -----------
skills = st.text_area("Enter your skills")

if st.button("Predict from Skills"):
    if skills.strip() != "":
        result, confidence, top_roles = predict_job(skills)

        st.success(f"✅ Best Role: {result}")
        st.write(f"Confidence: {confidence:.2f}")

        st.write("### 🔝 Top 3 Roles")
        for role, score in top_roles:
            st.write(f"{role} ({score:.2f})")
    else:
        st.warning("Enter skills first")

# ----------- PDF UPLOAD -----------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

if uploaded_file:
    resume_text = extract_text(uploaded_file)

    st.subheader("Extracted Resume Text")
    st.text_area("", resume_text, height=200)

    if st.button("Predict from Resume"):
        result, confidence, top_roles = predict_job(resume_text)

        st.success(f"✅ Best Role: {result}")
        st.write(f"Confidence: {confidence:.2f}")

        st.write("### 🔝 Top 3 Roles")
        for role, score in top_roles:
            st.write(f"{role} ({score:.2f})")

# ----------- FOOTER -----------
st.markdown("---")
st.markdown("🚀 Built with Streamlit")
st.markdown("🚀 build by srikanth-52")