# 🛡️ # ATMDefendX – CV-Powered ATM Tampering Detection Framework

This project simulates and detects ATM tampering activities like card skimming, hidden cameras, fake keypads, and stickers using computer vision techniques and a Siamese neural network.

---

## 📌 Features

- Capture or upload ATM images via a Streamlit app
- Siamese network compares suspect vs. clean images
- Tampering type inference from image filename
- LLM-based explanation for tampering detection
- JSON and PDF alert report downloads
- Simulated data generation using OpenCV
- Supports webcam integration for live capture

---

## 🧠 Model

We use a Siamese neural network with ResNet-18 as the base encoder to compare two ATM images and predict similarity.

- If similarity score >= 0.5 → **Tampered**
- Else → **Clean**

---

## 📦 Directory Structure

```
ATM_Fraud_Visual_Agent/
│
├── app/                     # Streamlit app
├── data/
│   ├── clean_atms/         # Clean ATM images (downloaded)
│   └── tampered_atms/      # Tampered ATM images (simulated)
├── models/
│   └── siamese_atm_model.pth  # Trained Siamese model
├── agents/
│   ├── comparator_agent.py
│   └── llm_agent.py
├── utils/
│   └── tamper_generator.py  # Tampering simulation with OpenCV
├── README.md
```

---

## 🏗️ Setup Instructions

1. Create virtual environment:
    ```bash
    python -m venv env_Atm
    source env_Atm/bin/activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit UI:
    ```bash
    streamlit run app/streamlit_ui.py
    ```

---

## 🎮 Demo Features

- Webcam support for suspect ATM image
- Tampering tags like:
  - `🔢 Keypad Overlay`
  - `📷 Hidden Camera`
  - `💳 Fake Card Reader`
  - `🏷️ Security Sticker`
- LLM Reasoning (via LM Studio - Mythomax 13B)
- PDF/JSON alert generation

---

## 📚 Model Training

To retrain Siamese model:
```bash
python train_siamese.py
```

It loads pairs of clean vs tampered images, uses ResNet18 embeddings, and trains a binary classifier.

---

## ⚠️ Tamper Simulation Types

- Fake Reader: Simulated via a black rectangle
- Hidden Camera: Red circle dot in frame corners
- Overlay Sticker: Yellow warning patch
- Keypad Overlay: Grey box around keypad area

---

## 🔒 Disclaimer

This is a simulation for research and prototyping only. No real ATM data was used.

---

## 👨‍💻 Author

**Gobinath Subramani** – Senior AI Consultant