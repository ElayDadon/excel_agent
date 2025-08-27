# app.py
import streamlit as st
import pandas as pd
from agent_logic import create_gemini_client, get_response_from_gemini, execute_code

# --- The CORRECT and SECURE way to get the API key for deployment ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

st.set_page_config(layout="wide", page_title="Agent ניתוח נתונים")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = None

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

if not GEMINI_API_KEY:
    st.error("שגיאה: מפתח ה-API של Gemini אינו מוגדר בהגדרות האפליקציה.")
elif st.session_state.dataframe is None:
    st.info("אנא העלה קובץ אקסל כדי להתחיל את השיחה.")
else:
    if st.session_state.gemini_client is None:
        st.session_state.gemini_client = create_gemini_client(GEMINI_API_KEY)
    if st.session_state.gemini_client is None:
        st.error("מפתח ה-API שהוגדר אינו תקין או שאירעה שגיאה.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["type"] == "data":
                    st.dataframe(message["content"])
                elif message["type"] == "plot":
                    st.plotly_chart(message["content"])
                else:
                    st.markdown(message["content"])
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