# app.py
import streamlit as st
import pandas as pd
from agent_logic import create_gemini_client, get_response_from_gemini, execute_code

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
# ננסה ליצור את ה-client רק פעם אחת
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