# 🌐 TraCare – Personalised Diabetes Assistant  

> 🩺 *Your multilingual buddy for smarter, personalised diabetes care.*  

[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)]()  
[![Python](https://img.shields.io/badge/Backend-Python-blue?logo=python)]()  
[![AI](https://img.shields.io/badge/AI-Google%20Gemini-green?logo=google)]()  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

---

## 📖 Table of Contents  
- [🚀 About the Project](#-about-the-project)  
- [⚙️ Features](#️-features)  
- [🛠️ Tech Stack](#️-tech-stack)  
- [🔄 How It Works](#-how-it-works-step-by-step)  
- [💡 Innovation & Creativity](#-innovation--creativity)  
- [🔮 Future Scope](#-future-scope)  
- [📷 Demo & Screenshots](#-demo--screenshots)  
- [📌 Installation & Setup](#-installation--setup)  
- [📂 Project Structure](#-project-structure)  
- [👨‍⚕️ Authors](#-authors)  
- [📜 License](#-license)  

---

## 🚀 About the Project  

Over **77 million Indians** live with diabetes — yet most health apps:  
❌ Forget past inputs  
❌ Work only in **English**  
❌ Lack **personalisation**  

✨ **TraCare** changes this by being a **Streamlit-based AI chatbot** that:  
- 🗣️ Speaks English + **multiple Indian languages**  
- 🧠 Remembers patient details (age, diabetes type, meds, symptoms, etc.)  
- ⚕️ Generates **India-specific, diabetes-safe responses**  
- 📊 Predicts **glucose spikes** with an integrated ML model  
- 📑 Stores logs & generates **medical summary reports**  

💬 **Vision**: *“Smart care that remembers you.”*  

---

## ⚙️ Features  

- 🌍 **Multilingual Conversations** – Supports multiple Indian languages via **SUTRA-v2**  
- 🧠 **Personalised Memory Layer (Mem0)** – Stores user profile & chat history  
- 🤖 **Generative AI Responses** – Contextual & diabetes-safe answers using **Google Gemini**  
- 🍛 **Food & Glycemic Index Analysis** – Uses **Edamam API** for Indian dietary habits  
- 🎙️ **Speech Support** – STT via **Assembly AI** (English, more languages in future)  
- 📑 **Medical Summary Reports** – Food, exercise, glucose logs for doctors/patients  
- 📈 **Prediction Model** – Forecasts hypoglycemia spikes using diet, exercise & CGM data  
- ⌚ **Google Fit Integration** – Syncs vitals to improve recommendations  

---

## 🛠️ Tech Stack  

| Layer        | Technology |
|--------------|------------|
| **Frontend** | Streamlit |
| **Backend**  | Python |
| **LLMs**     | Google Gemini, SUTRA-v2 |
| **Memory**   | Mem0 |
| **APIs**     | Edamam API, Google Fit API |
| **Speech**   | Assembly AI (STT) |
| **ML Model** | Hypoglycemia spike prediction |

---

## 🔄 How It Works (Step-by-Step)  

<details>
<summary>📝 Expand Workflow</summary>  

1️⃣ Enter **SUTRA & Mem0 API keys**  
2️⃣ Choose preferred **language**  
3️⃣ If **new user** → collects diabetes details (age, type, medication, lifestyle) → stores in Mem0  
4️⃣ Each query is **translated (if needed)** → processed → logged  
5️⃣ **Gemini** generates diabetes-safe responses  
6️⃣ Responses are **translated back** (if needed) and displayed in UI  
7️⃣ Past conversations & health data are stored for **future recall**  
8️⃣ Voice model converts response text → **audio playback**  
9️⃣ Reports generated on **food, exercise, glucose values, and vitals**  

</details>  

---

## 💡 Innovation & Creativity  

- 🗣️ **Persistent Multilingual Personalisation** – remembers your language & context  
- 🍛 **Cultural Context Awareness** – dietary recommendations tailored to Indian foods  
- 📈 **Integrated ML Model** – predicts hypoglycemia spikes before they happen  
- ⌚ **Google Fit Integration** – health data synced directly for smarter advice  

---

## 🔮 Future Scope  

- 📱 **Android App Development** – for wider accessibility  
- ⌚ **IoT Integration** – sync with CGM monitors & Fitbit devices  
- 🌍 **More Languages** – expand TTS/STT support beyond English & Hindi  

---

## 📷 Demo & Screenshots  

👉 *(Replace with actual images/videos once available)*  

- 🖼️ **Homepage UI**  
![Homepage Screenshot](docs/screenshots/home.png)  

- 💬 **Chat Interface**  
![Chat Screenshot](docs/screenshots/chat.png)  

- 📊 **Medical Report Generation**  
![Report Screenshot](docs/screenshots/report.png)  

---

## 📌 Installation & Setup  

```bash
# Clone the repository
git clone https://github.com/<your-username>/TraCare.git
cd TraCare

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
