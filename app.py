# app.py (גרסה מאוחדת)
import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.generativeai import types

# =======================================================================
# ====> כל הלוגיקה נמצאת עכשיו כאן
# =======================================================================

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
        error_message = str(e)
        return None, error_message

def get_response_from_gemini(client, dataframe, query, chat_history):
    """שולח בקשה ל-Gemini באמצעות אובייקט ה-Client."""
    if not client:
        return "Error: Gemini client is not initialized."

    model = "gemini-2.0-flash"
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

# =======================================================================
# ====> ממשק המשתמש של Streamlit מתחיל כאן
# =======================================================================

# --- קריאה מאובטחת של המפתח מ-Streamlit Secrets ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

st.set_page_config(layout="wide", page_title="Agent ניתוח נתונים")

# אתחול משתנים
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = None
if "client_error" not in st.session_state:
    st.session_state.client_error = None

st.sidebar.title("הגדרות")
st.sidebar.header("העלאת קובץ Excel")
uploaded_file = st.sidebar.file_uploader("בחר קובץ", type=["xlsx", "xls"])

if uploaded_file is not None:
    if st.session_state.dataframe is None:
        with st.spinner("טוען את הנתונים..."):
            st.session_state.dataframe = pd.read_excel(uploaded_file)
            st.session_state.messages = []
            st.sidebar.success("הקובץ נטען בהצלחה!")

st.title("🤖 Agent מקצועי לניתוח נתונים")
st.write("העלה קובץ אקסל בצד ימין, ואז התחל לשאול שאלות על הנתונים שלך.")

# --- לוגיקת אימות API משודרגת עם הצגת שגיאות ---
if st.session_state.gemini_client is None and st.session_state.client_error is None:
    if not GEMINI_API_KEY:
        st.session_state.client_error = "שגיאה: מפתח ה-API של Gemini אינו מוגדר ב-Secrets של האפליקציה."
    else:
        client, error_message = create_gemini_client(GEMINI_API_KEY)
        if error_message:
            st.session_state.client_error = f"אירעה שגיאה באימות מול Google:\n\n```\n{error_message}\n```"
        else:
            st.session_state.gemini_client = client

# --- תצוגה ראשית ---
if st.session_state.client_error:
    st.error(st.session_state.client_error)
elif st.session_state.dataframe is None:
    st.info("אנא העלה קובץ אקסל כדי להתחיל את השיחה.")
else:
    # הצגת היסטוריית הצ'אט
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "data":
                st.dataframe(message["content"])
            elif message["type"] == "plot":
                st.plotly_chart(message["content"])
            else:
                st.markdown(message["content"])
    
    # קבלת קלט מהמשתמש
    if prompt := st.chat_input("שאל אותי כל דבר על הנתונים..."):
        st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("חושב..."):
                gemini_response = get_response_from_gemini(st.session_state.gemini_client, st.session_state.dataframe, prompt, st.session_state.messages)
                result = execute_code(st.session_state.dataframe, gemini_response)
                if result["type"] == "data":
                    st.dataframe(result["content"])
                elif result["type"] == "plot":
                    st.plotly_chart(result["content"])
                else:
                    st.markdown(result["content"])
                st.session_state.messages.append({"role": "assistant", **result})