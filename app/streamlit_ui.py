"""
ATM Fraud Visual Agent – Streamlit Interface

This application allows users to either upload or capture images of ATMs
to visually detect tampering (e.g., skimming devices, hidden cameras).
It compares a suspect ATM image against a known clean reference using
a Siamese neural network, tags likely tampering types, and provides a 
diagnostic report via an LLM powered by LM Studio.

Features:
- Upload clean & suspect ATM images
- Webcam capture support
- Siamese Network similarity scoring
- Scenario tagging via filename heuristic
- LLM reasoning for detected tampering
- Alert report downloads (JSON + PDF)
"""

import streamlit as st  # UI framework
import os               # For path operations
import sys              # To modify import path
import json             # For report export
from fpdf import FPDF   # For PDF report generation
import io               # For in-memory byte streams
from PIL import Image   # For image handling

# === Local imports for custom agents ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.comparator_agent import ComparatorAgent
from agents.llm_agent import call_llm_diagnostic

# === Page setup ===
st.set_page_config(page_title="ATM Fraud Visual Agent", layout="centered")
st.title("🛡️ ATM Fraud Visual Agent")
st.markdown("Upload or capture ATM images to detect tampering.")

# === Optional Webcam Capture ===
st.markdown("### 📷 Capture Suspect ATM Image (Webcam)")
enable_cam = st.checkbox("📷 Enable Webcam")  # Toggle webcam capture

camera_image = None
if enable_cam:
    camera_image = st.camera_input("Capture Image")  # Activate webcam capture UI

test_img = None
test_img_bytes = None
test_image_pil = None  # For PIL image if captured

# === Process webcam image if captured ===
if camera_image:
    try:
        image_bytes = camera_image.getvalue()
        if len(image_bytes) > 0:
            image_stream = io.BytesIO(image_bytes)
            test_image_pil = Image.open(image_stream).convert("RGB")
            test_image_pil.save("temp_test.jpg")

            # Store as byte stream for UI preview
            img_bytes = io.BytesIO()
            test_image_pil.save(img_bytes, format="JPEG")
            test_img_bytes = img_bytes.getvalue()

            st.success("📸 Test image captured")
            test_img = open("temp_test.jpg", "rb")
        else:
            st.warning("⚠️ Camera captured empty image. Try again.")
    except Exception as e:
        st.error(f"❌ Error decoding image: {e}")

# === Upload images ===
ref_img = st.file_uploader("📥 Upload Clean Reference ATM Image", type=["jpg", "jpeg", "png"])
if not test_img:
    test_img = st.file_uploader("📥 Upload Suspect ATM Image", type=["jpg", "jpeg", "png"])

# === If both images are ready ===
if ref_img and test_img:
    # Save reference image locally
    with open("temp_ref.jpg", "wb") as f:
        f.write(ref_img.read())

    # If not from webcam, save test image as well
    if not test_image_pil:
        with open("temp_test.jpg", "wb") as f:
            f.write(test_img.read())

    # === Show uploaded images side-by-side ===
    col1, col2 = st.columns(2)
    with col1:
        st.image("temp_ref.jpg", caption="Clean Reference", use_container_width=True)
    with col2:
        if test_img_bytes:
            st.image(test_img_bytes, caption="Test Image", use_container_width=True)
        else:
            st.image("temp_test.jpg", caption="Test Image", use_container_width=True)

    # === Run visual comparison ===
    if st.button("🔍 Run Visual Comparison"):
        comparator = ComparatorAgent()

        # Handle webcam image (PIL) vs file upload
        if test_image_pil:
            score, result = comparator.compare("temp_ref.jpg", test_image_pil)
        else:
            score, result = comparator.compare("temp_ref.jpg", "temp_test.jpg")

        st.success(f"**Similarity Score:** {score:.4f} → **{result.upper()}**")

        # === Tag scenario based on filename keywords ===
        def infer_scenario(filename):
            name = filename.lower()
            tags = []
            if "keypad" in name:
                tags.append("🔢 Keypad Overlay")
            if "camera" in name:
                tags.append("📷 Hidden Camera")
            if "sticker" in name:
                tags.append("🏷️ Fake Sticker")
            if "reader" in name:
                tags.append("💳 Fake Card Reader")
            return tags or ["Unknown Tampering"]

        scenario_tags = infer_scenario(getattr(test_img, 'name', 'temp_test.jpg'))

        st.markdown("### 🏷️ Detected Tampering Types:")
        st.markdown(", ".join(scenario_tags))

        # === LLM Reasoning ===
        if result == "Tampered":
            with st.spinner("🧠 Calling LLM for diagnostic reasoning..."):
                llm_response = call_llm_diagnostic(score, result)
                st.markdown("### 🤖 LLM Diagnostic Reasoning")
                st.markdown(llm_response)
        else:
            llm_response = "No reasoning – image marked clean."

        # === Downloadable JSON and PDF Alert Report ===
        st.markdown("### 📄 Download Alert Report")

        # Prepare structured report
        report_data = {
            "reference_image": getattr(ref_img, 'name', 'temp_ref.jpg'),
            "test_image": getattr(test_img, 'name', 'temp_test.jpg'),
            "similarity_score": round(score, 4),
            "prediction": result,
            "tags": scenario_tags,
            "llm_reasoning": llm_response
        }

        # JSON download
        json_str = json.dumps(report_data, indent=2)
        st.download_button("⬇️ Download JSON", json_str, file_name="atm_alert_report.json", mime="application/json")

        # PDF generation with Unicode-safe encoding
        def safe_text(text):
            return str(text).encode('latin-1', 'replace').decode('latin-1')

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for k, v in report_data.items():
            if isinstance(v, list):
                v = ", ".join(v)
            sanitized = safe_text(f"{k.capitalize()}: {v}")
            pdf.multi_cell(0, 10, sanitized)

        pdf_path = "atm_alert_report.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name="atm_alert_report.pdf", mime="application/pdf")

else:
    st.info("👈 Please upload both images (or capture the test image using your webcam) to begin.")