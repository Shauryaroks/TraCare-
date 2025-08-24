import os
import json
import requests
import streamlit as st
import google.generativeai as genai
from mem0 import MemoryClient
from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env into environment
from google_fit import authenticate_google_fit, fetch_step_count, fetch_token, get_auth_url
import tomli

with open("config.toml", "rb") as f:
    config = tomli.load(f)

# --------------------------
# AssemblyAI for transcription
# --------------------------
import assemblyai as aai

ASSEMBLY_API_KEY = config["ASSEMBLYAI_API_KEY"]
aai.settings.api_key = ASSEMBLY_API_KEY
SCOPES = ['https://www.googleapis.com/auth/fitness.activity.read']


def transcribe_audio(audio_bytes) -> str:
    """Send audio bytes to AssemblyAI and return transcript text."""
    transcriber = aai.Transcriber()
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    transcript = transcriber.transcribe("temp_audio.wav")
    return transcript.text if transcript and transcript.text else ""


# --------------------------
# Diabetes Health Assistant
# --------------------------
class DiabetesHealthAssistant:
    def __init__(self):
        self.languages = [
            "English", "Hindi", "Gujarati", "Bengali", "Tamil",
            "Telugu", "Kannada", "Malayalam", "Punjabi", "Marathi",
            "Urdu", "Assamese", "Odia", "Sanskrit"
        ]
        self.language_codes = {
            "English": "en", "Hindi": "hi", "Gujarati": "gu", "Bengali": "bn",
            "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Malayalam": "ml",
            "Punjabi": "pa", "Marathi": "mr", "Urdu": "ur", "Assamese": "as",
            "Odia": "or", "Sanskrit": "sa"
        }
        self.sutra_base_url = "https://api.two.ai/v2/chat/completions"

    def initialize_apis(self):
        try:
            genai.configure(api_key=config["GENAI_KEY"])
            self.gemini = genai.GenerativeModel("gemini-1.5-flash")
            self.gemini_logs = genai.GenerativeModel("gemini-1.5-pro")
            self.client = MemoryClient(api_key=config["MEM0_KEY"])
            self.sutra_api_key = config["SUTRA_KEY"]
            return True
        except Exception as e:
            st.error(f"API init failed: {e}")
            return False

    def call_sutra_api(self, prompt, max_tokens=1000):
        headers = {
            "Authorization": f"Bearer {self.sutra_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "sutra-v2",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        response = requests.post(self.sutra_base_url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"API error: {response.status_code} {response.text}")

    def translate_text(self, text, target_language):
        if target_language == "English":
            return text
        prompt = f"Translate to {target_language}: {text}"
        return self.call_sutra_api(prompt, max_tokens=200)

    def get_chat_response(self, query, user_id, selected_language):
        query_en = query if selected_language == "English" else self.call_sutra_api(
            f"Translate this to English: {query}", max_tokens=200
        )
        search_results = self.client.search(query_en, user_id=user_id)
        mem0_memories = [r.get("message", {}).get("content", r.get("memory", "")) for r in search_results]
        context = "\n".join(mem0_memories)

        # Include Google Fit data if available
        fit_context = ""
        if st.session_state.get("fit_creds") and st.session_state.get("fit_steps") is not None:
            fit_context = f"\nUser's recent step count: {st.session_state.fit_steps} steps"

        prompt = f"""
        You are a diabetes-friendly AI assistant.
        Context from user memory:
        {context}
        {fit_context}

        Query: {query_en}
        Respond in {selected_language}.
        """
        response = self.gemini.generate_content(prompt).text
        if selected_language != "English":
            response = self.translate_text(response, selected_language)

        self.client.add([{ "role": "user", "content": query},
                         { "role": "assistant", "content": response}], user_id=user_id)
        return response

    def log_health_data(self, user_id, exercise, diet, sugar_level):
        log_entry = f"Exercise: {exercise}\nDiet: {diet}\nSugar Level: {sugar_level}"
        self.client.add([{ "role": "system", "content": log_entry}], user_id=user_id)
        return log_entry


# --------------------------
# Streamlit App
# --------------------------
def main():
    st.set_page_config(page_title="Diabetes Health Assistant", page_icon="🩺", layout="wide")
    st.title("🩺 Diabetes Health Assistant")
    st.caption("Your personalized diabetes management companion")

    if "assistant" not in st.session_state:
        st.session_state.assistant = DiabetesHealthAssistant()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "language" not in st.session_state:
        st.session_state.language = "English"

    # Sidebar for setup
    with st.sidebar:
        st.header("🔑 API Configuration")
        language = st.selectbox("Select Language", st.session_state.assistant.languages, index=0)
        if st.button("Initialize"):
            ok = st.session_state.assistant.initialize_apis()
            if ok:
                st.session_state.language = language
                st.success("APIs initialized!")

        # --------------------------
        # Google Fit Auth (only show after APIs init)
        # --------------------------
        if getattr(st.session_state.assistant, "client", None):
            st.markdown("---")
            st.header("🏃 Google Fit Integration")

            # Load existing credentials if available
            if "fit_creds" not in st.session_state:
                st.session_state.fit_creds = authenticate_google_fit()
            if "fit_steps" not in st.session_state:
                st.session_state.fit_steps = None

            if not st.session_state.fit_creds:
                # Check for authorization code in query params (after redirect)
                if "code" in st.query_params:
                    auth_code = st.query_params["code"][0]
                    try:
                        creds = fetch_token(auth_code)
                        st.session_state.fit_creds = creds
                        with open('token.json', 'w') as token:
                            token.write(creds.to_json())
                        del st.query_params["code"]
                        st.success("Google Fit connected successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Authorization failed: {e}")
                        del st.query_params["code"]
                else:
                    # Show authorization link
                    auth_url = get_auth_url()
                    st.markdown(f"**[🔗 Click here to authorize Google Fit access]({auth_url})**")
                    st.info("After authorizing, you will be redirected back to this app.")
            else:
                st.success("✅ Google Fit connected")
                if st.button("Fetch Step Data", key="fetch_steps"):
                    try:
                        steps = fetch_step_count(st.session_state.fit_creds, days=1)
                        st.session_state.fit_steps = steps
                        st.success(f"Today's steps: {steps}")
                    except Exception as e:
                        st.error(f"Failed to fetch steps: {e}")
                
                if st.button("Disconnect Google Fit", key="disconnect_fit"):
                    if os.path.exists('token.json'):
                        os.remove('token.json')
                    st.session_state.fit_creds = None
                    st.session_state.fit_steps = None
                    st.success("Google Fit disconnected")
                    st.rerun()


    if not getattr(st.session_state.assistant, "client", None):
        st.warning("Enter API keys in sidebar to continue.")
        return

    # User login/registration
    if not st.session_state.user_id:
        st.subheader("👤 User Login")
        user_id = st.text_input("Enter your unique username/ID")
        if st.button("Login"):
            if user_id.strip():
                st.session_state.user_id = user_id.strip()
                st.success(f"Logged in as {st.session_state.user_id}")
            else:
                st.error("Enter a valid ID.")
        return

    # --------------------------
    # Display Google Fit data if available
    # --------------------------
    if st.session_state.get("fit_steps") is not None:
        st.sidebar.markdown("---")
        st.sidebar.metric("📊 Today's Steps", st.session_state.fit_steps)

    # --------------------------
    # Logging UI
    # --------------------------
    st.subheader("📊 Log Your Health Data")
    with st.form("log_form"):
        exercise = st.text_input("Exercise (e.g., 30 min jogging, 100 pushups)")
        diet = st.text_input("Diet (e.g., 2 chapatis, rice, dal)")
        sugar_level = st.number_input("Sugar Level (mg/dL)", min_value=50, max_value=400, step=1)
        submitted = st.form_submit_button("Log Data")
        if submitted:
            entry = st.session_state.assistant.log_health_data(
                st.session_state.user_id, exercise, diet, sugar_level
            )
            st.success("Health data logged!")


    # --------------------------
    # Chat + Voice
    # --------------------------
    st.subheader("💬 Chat with Assistant")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = None

    # Option 1: text input
    query = st.chat_input("Type your question...")

    # Option 2: voice input
    audio_bytes = st.audio_input("🎤 Speak your query")
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        if st.button("Transcribe & Send"):
            with st.spinner("Transcribing..."):
                query = transcribe_audio(audio_bytes.getvalue())
            st.success(f"You said: {query}")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.assistant.get_chat_response(
                    query,
                    st.session_state.user_id,
                    st.session_state.language
                )
                st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
