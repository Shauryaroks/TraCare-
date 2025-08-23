import requests
import pandas as pd
import google.generativeai as genai
import io
from PIL import Image


class DiabeticMealAnalyzer:
    def __init__(self, edamam_app_id=None, edamam_app_key=None,
                 gemini_key=None, gemini_model="gemini-1.5-flash"):
        self.edamam_app_id = edamam_app_id
        self.edamam_app_key = edamam_app_key
        self.gemini_model = gemini_model
        self.gemini_configured = False

        if gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                self.gemini_configured = True
            except Exception as e:
                print(f"[Warning] Gemini config failed: {e}")

    # -------------------------
    # Gemini Vision
    # -------------------------
    def extract_food_names_with_gemini(self, image_bytes):
        """Identify foods in an image using Gemini Vision."""
        if not self.gemini_configured:
            return []

        try:
            model = genai.GenerativeModel(self.gemini_model)
            response = model.generate_content(
                ["Identify the main Indian foods in this image. "
                 "Return only a short comma-separated list of food names."],
                image=image_bytes
            )
            text = response.text or ""
            items = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]

            # Deduplicate
            seen, foods = set(), []
            for it in items:
                lower = it.lower()
                if lower not in seen:
                    seen.add(lower)
                    foods.append(it)
            return foods
        except Exception as e:
            print(f"[Error] Gemini Vision: {e}")
            return []

    @staticmethod
    def image_to_bytes(pil_image: Image.Image):
        """Convert PIL image to bytes for Gemini Vision."""
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    # -------------------------
    # Edamam Nutrition Lookup
    # -------------------------
    def analyze_meal_edamam(self, food_text):
        """Fetch nutrition info from Edamam API for one food."""
        url = "https://api.edamam.com/api/food-database/v2/parser"
        params = {
            "app_id": self.edamam_app_id,
            "app_key": self.edamam_app_key,
            "ingr": food_text
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {"Food": food_text, "Calories": "N/A", "Protein (g)": "N/A",
                    "Fat (g)": "N/A", "Carbs (g)": "N/A", "Fiber (g)": "N/A",
                    "Notes": f"API error: {e}"}

        food = None
        if data.get("parsed"):
            food = data["parsed"][0]["food"]
        elif data.get("hints"):
            food = data["hints"][0]["food"]

        if not food:
            return {"Food": food_text, "Calories": "N/A", "Protein (g)": "N/A",
                    "Fat (g)": "N/A", "Carbs (g)": "N/A", "Fiber (g)": "N/A",
                    "Notes": "Not found in Edamam"}

        nutrients = food.get("nutrients", {})
        return {
            "Food": food_text,
            "Calories": nutrients.get("ENERC_KCAL", "N/A"),
            "Protein (g)": nutrients.get("PROCNT", "N/A"),
            "Fat (g)": nutrients.get("FAT", "N/A"),
            "Carbs (g)": nutrients.get("CHOCDF", "N/A"),
            "Fiber (g)": nutrients.get("FIBTG", "N/A"),
            "Notes": ""
        }

    # -------------------------
    # Diabetes Insights
    # -------------------------
    def diabetes_insights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add diabetic insights + risk levels to DataFrame."""
        high_gi_foods = ["white rice", "rice", "potato", "white bread",
                         "sugar", "samosa", "gulab jamun", "paratha"]
        suggestions = {
            "rice": "Try brown rice or millets like ragi/jowar instead.",
            "potato": "Replace with sweet potato or non-starchy veggies.",
            "bread": "Choose whole wheat roti or multigrain bread.",
            "samosa": "Try baked samosas or steamed snacks like idli/dhokla.",
            "gulab jamun": "Choose fruit salad or sugar-free kheer.",
            "paratha": "Use less oil or choose whole-wheat roti."
        }

        insights, risks = [], []

        for _, row in df.iterrows():
            food_lower = str(row.get("Food", "")).lower()
            carbs = row.get("Carbs (g)", "N/A")
            comment_parts, risk = [], "Safe"

            try:
                if carbs != "N/A" and carbs not in (None, "") and float(carbs) > 30:
                    comment_parts.append("⚠️ High in carbs – monitor portion size.")
                    risk = "Caution"
            except:
                pass

            for gi in high_gi_foods:
                if gi in food_lower:
                    comment_parts.append("⚠️ High glycemic index – avoid or limit.")
                    if gi in suggestions:
                        comment_parts.append("👉 " + suggestions[gi])
                    risk = "Avoid"

            if not comment_parts:
                comment_parts.append("✅ Generally safe for diabetics in moderate portions.")

            insights.append(" ".join(comment_parts))
            risks.append(risk)

        df["Diabetes Insights"] = insights
        df["Risk Level"] = risks
        return df

    # -------------------------
    # Full Analysis
    # -------------------------
    def analyze_meal(self, foods: list) -> pd.DataFrame:
        """Analyze a list of foods: nutrition + diabetic insights."""
        results = [self.analyze_meal_edamam(f) for f in foods]
        df = pd.DataFrame(results)
        return self.diabetes_insights(df)
