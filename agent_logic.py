# agent_logic.py
import google.generativeai as genai
from google.generativeai import types
import pandas as pd

def create_gemini_client(api_key):
    """
    יוצר אובייקט Client של Gemini.
    במקרה של הצלחה, מחזיר (client, None).
    במקרה של כישלון, מחזיר (None, error_message_string).
    """
    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        # המר את אובייקט השגיאה למחרוזת טקסט ברורה
        error_message = str(e)
        return None, error_message

def get_response_from_gemini(client, dataframe, query, chat_history):
    """שולח בקשה ל-Gemini באמצעות אובייקט ה-Client."""
    if not client:
        return "Error: Gemini client is not initialized."

    model = "gemini-1.5-flash-latest"
    df_columns = dataframe.columns.tolist()
    prompt_text = f"""
    You are a world-class, friendly, and conversational data analyst AI.
    Your main goal is to help a user understand their data by answering questions in natural Hebrew.
    You specialize in Python with pandas and Plotly.

    **CONTEXT:**
    - You have access to a pandas DataFrame named `df` with the following columns: {df_columns}.
    - The user's current query is: "{query}"
    - The recent conversation history is: {chat_history}

    **YOUR CORE PRINCIPLES:**
    1.  **Be Conversational and Proactive:** When first asked "what is in the file" (e.g., "מה יש בקובץ"), provide a comprehensive text-only overview.
    2.  **Understand User Intent & Handle Typos:** Your top priority is to understand what the user *means*, not just what they typed. (e.g., "קבוץ" -> "קובץ").
    3.  **Handle Ambiguity:** If a query is vague like "תספק לי", use the immediate preceding context.

    **RESPONSE FORMATTING (Strictly follow this):**
    - To generate pandas code for data/analysis, wrap it in ```python\\n[CODE]\\n...```
    - To generate Plotly code for a graph, wrap it in ```python\\n[PLOT]\\n...```
    - For conversational text answers, respond directly without any wrappers.
    """

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])]

    try:
        response = client.models.generate_content(model=f"models/{model}", contents=contents)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"

def execute_code(dataframe, gemini_response):
    """מריץ את הקוד שהתקבל באופן בטוח."""
    response_text = gemini_response.strip().replace("`", "")
    if "python\\n[CODE]" in response_text:
        code_to_run = response_text.split("python\\n[CODE]")[1].strip()
        try:
            result = eval(code_to_run, {"df": dataframe, "pd": pd})
            return {"type": "data", "content": result}
        except Exception as e:
            return {"type": "error", "content": f"שגיאה בהרצת קוד הנתונים: {e}\\nקוד: {code_to_run}"}
    elif "python\\n[PLOT]" in response_text:
        code_to_run = response_text.split("python\\n[PLOT]")[1].strip()
        try:
            local_scope = {"df": dataframe}
            exec(code_to_run, globals(), local_scope)
            fig = local_scope.get('fig', None)
            if fig:
                return {"type": "plot", "content": fig}
            else:
                return {"type": "error", "content": "קוד הגרף לא יצר משתנה בשם 'fig'."}
        except Exception as e:
            return {"type": "error", "content": f"שגיאה ביצירת הגרף: {e}\\nקוד: {code_to_run}"}
    else:
        return {"type": "text", "content": response_text}