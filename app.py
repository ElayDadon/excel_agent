# app.py — Streamlit + Gemini (new SDK) Excel Q&A in Hebrew
# ---------------------------------------------------------
# Requirements (requirements.txt):
# streamlit
# pandas
# openpyxl
# plotly
# google-genai>=1.30.0
#
# Secrets:
#   In Streamlit Community Cloud, set:  [GEMINI_API_KEY = "your_api_key"]

import os
import io
import re
import json
import textwrap
import traceback
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# NEW SDK — do NOT use `google.generativeai`
from google import genai
from google.genai import types as genai_types


# -------------------------- Streamlit page setup -----------------------------

st.set_page_config(
    page_title="Excel Q&A (Hebrew) — Gemini",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Excel Q&A (Hebrew) — Gemini 2.5 Flash")
st.caption(
    "העלו קובץ אקסל, כתבו שאלה בעברית — והמודל ייצור קוד פייתון (pandas/plotly) שמריץ ומציג תשובה."
)


# ------------------------------ Utilities -----------------------------------

def get_api_key() -> Optional[str]:
    """Get Gemini API key from Streamlit secrets or environment."""
    key = None
    try:
        # st.secrets raises if not configured; guard with try
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not key:
        key = os.environ.get("GEMINI_API_KEY")
    return key


@st.cache_resource(show_spinner=False)
def get_client() -> genai.Client:
    api_key = get_api_key()
    if not api_key:
        st.stop()
    return genai.Client(api_key=api_key)


def detect_local_google_package():
    """Warn if a local 'google' folder shadows the real Google namespace packages."""
    if os.path.isdir("google") and not os.path.exists(os.path.join("google", "__init__.py")):
        st.warning(
            "נראה שיש בתיקייה של הפרויקט תיקייה בשם **google/**. זה עלול לשבור את הייבוא של ה-SDK. "
            "מומלץ לשנות את שם התיקייה.",
            icon="⚠️",
        )


def df_schema_and_sample(df: pd.DataFrame, max_rows: int = 5, max_len: int = 120) -> Dict[str, Any]:
    """Build a compact schema + sample to share with the model (no full data)."""
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample_df = df.head(max_rows).copy()
    for col in sample_df.columns:
        if sample_df[col].dtype == object:
            sample_df[col] = sample_df[col].astype(str).str.slice(0, max_len)
    return {"schema": schema, "sample_rows": sample_df.to_dict(orient="records")}


def extract_code_from_markdown(md: str) -> Optional[str]:
    """
    Extract first Python code block from a markdown string.
    Accepts ```python ...``` or ``` ... ```.
    """
    fence = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    m = fence.search(md or "")
    return m.group(1).strip() if m else None


def make_instruction_prompt() -> str:
    """System-style instruction for code generation."""
    return textwrap.dedent('''
    You are a senior Python data engineer. Generate **only Python code** (no prose)
    that defines a single function with the exact signature:

        def solve(df: "pandas.DataFrame", question: str):
            """
            df: the user's DataFrame (from Excel)
            question: the user's Hebrew question about the data
            Returns one of:
              - a plain string with the answer/summary, OR
              - a pandas.DataFrame to display, OR
              - a plotly Figure, OR
              - a tuple like (string|None, figure|dataframe|list_of_figures|list_of_dataframes)
            """

    Rules:
    - Use ONLY pandas and plotly (plotly.express as px, graph_objects as go). No other libs.
    - DO NOT read or write files. DO NOT access network or environment.
    - Assume df is clean but handle common issues gracefully (missing values, types).
    - Prefer vectorized pandas ops. For group-by, filters, date handling use pandas idioms.
    - If the question implies a chart, return a plotly Figure (bar/line/pie/scatter/box/hist).
    - If a clear tabular answer exists, return a small pandas DataFrame.
    - If both text and a figure/table help: return (text, fig_or_df).
    - Keep charts readable: title, labels, sensible sorting/aggregation.
    - If the question is ambiguous, infer the most likely intent and state assumptions in the returned text.

    IMPORTANT:
    - The ONLY output must be valid Python code for the `solve` function.
    - Do NOT import anything at top-level; put imports inside the function if needed.
    ''')


def build_user_prompt(question_he: str, summary: Dict[str, Any]) -> str:
    """User prompt including schema+sample and the Hebrew question."""
    return textwrap.dedent(f'''
    נתוני עזר על הדאטה־פריים:
    - עמודות וסוגים: {json.dumps(summary["schema"], ensure_ascii=False)}
    - דוגמת שורות (מוגבל): {json.dumps(summary["sample_rows"], ensure_ascii=False)}

    שאלה בעברית:
    {question_he}

    החזר אך ורק קוד פייתון (solve) כמוסבר לעיל.
    ''')


def generate_solver_code(question: str, summary: Dict[str, Any], model_name: str, temperature: float) -> str:
    """Call Gemini (new SDK) to generate the solve(df, question) code."""
    client = get_client()

    system_msg = make_instruction_prompt()
    user_msg = build_user_prompt(question, summary)

    resp = client.models.generate_content(
        model=model_name,
        contents=[system_msg, user_msg],
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
        )
    )

    text = (resp.text or "").strip()
    code = extract_code_from_markdown(text) or text  # Sometimes the model returns raw code
    if "def solve(" not in code:
        raise ValueError("The generated content did not include a valid solve(df, question) function.")
    return code


def safe_exec_solve(code: str):
    """
    Execute the generated code in a constrained namespace and return the solve function.
    We avoid exposing dangerous builtins or modules.
    """
    safe_builtins = {
        "len": len, "range": range, "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "sorted": sorted, "enumerate": enumerate, "zip": zip, "any": any, "all": all, "map": map, "filter": filter
    }

    allowed_globals = {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
        "pd": pd, "px": px, "go": go,
    }

    local_ns: Dict[str, Any] = {}

    exec(code, allowed_globals, local_ns)  # noqa: S102 (controlled)
    solve = local_ns.get("solve") or allowed_globals.get("solve")
    if not callable(solve):
        raise ValueError("No callable 'solve' function was defined by the generated code.")
    return solve


def render_result(result: Any):
    """Render the solve() result in Streamlit."""
    if result is None:
        st.info("לא התקבלה תשובה.")
        return

    if isinstance(result, tuple) and len(result) == 2:
        text, payload = result
        if text:
            st.write(text)

        if payload is None:
            return

        if hasattr(payload, "to_plotly_json"):
            st.plotly_chart(payload, use_container_width=True)
            return

        if isinstance(payload, pd.DataFrame):
            st.dataframe(payload, use_container_width=True)
            return

        if isinstance(payload, list):
            for item in payload:
                if hasattr(item, "to_plotly_json"):
                    st.plotly_chart(item, use_container_width=True)
                elif isinstance(item, pd.DataFrame):
                    st.dataframe(item, use_container_width=True)
                else:
                    st.write(item)
            return

        st.write(payload)
        return

    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True)
        return

    if hasattr(result, "to_plotly_json"):
        st.plotly_chart(result, use_container_width=True)
        return

    if isinstance(result, list):
        for item in result:
            if hasattr(item, "to_plotly_json"):
                st.plotly_chart(item, use_container_width=True)
            elif isinstance(item, pd.DataFrame):
                st.dataframe(item, use_container_width=True)
            else:
                st.write(item)
        return

    st.write(str(result))


# ------------------------------- UI Controls --------------------------------

detect_local_google_package()

api_key = get_api_key()
if not api_key:
    st.error("❌ לא נמצא מפתח API של Gemini. הוסף GEMINI_API_KEY ב-Secrets של Streamlit או כמשתנה סביבה.")
    st.stop()

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    uploaded = st.file_uploader("העלו קובץ Excel", type=["xlsx", "xls", "xlsm", "csv"])
    question = st.text_area("שאלה בעברית על הנתונים", height=120, placeholder="לדוגמה: מה המכירות החודשיות הממוצעות לכל קטגוריה בשנת 2024?")
with col2:
    model_name = st.selectbox(
        "מודל",
        options=["gemini-2.5-flash-001", "gemini-2.0-flash-001"],
        index=0,
        help="מומלץ: gemini-2.5-flash-001"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    show_code = st.toggle("להציג את הקוד שנוצר", value=False)

st.divider()

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"קריאת הקובץ נכשלה: {e}")
        st.stop()

    st.subheader("תצוגה מקדימה של הנתונים")
    st.dataframe(df.head(100), use_container_width=True)

    meta = df_schema_and_sample(df)

    st.markdown("### תשובה לשאלה")
    run = st.button("הרץ ניתוח עם Gemini", type="primary", use_container_width=True)

    if run:
        if not question.strip():
            st.warning("אנא הזינו שאלה בעברית.")
            st.stop()

        with st.spinner("המודל מייצר קוד פייתון..."):
            try:
                code = generate_solver_code(
                    question=question.strip(),
                    summary=meta,
                    model_name=model_name,
                    temperature=temperature,
                )
            except Exception as e:
                st.error("שגיאה ביצירת הקוד מהמודל:")
                st.exception(e)
                st.stop()

        if show_code:
            st.markdown("#### הקוד שנוצר")
            st.code(code, language="python")

        with st.spinner("מריץ את הקוד על הדאטה..."):
            try:
                solve = safe_exec_solve(code)
                out = solve(df, question.strip())
                render_result(out)
            except Exception:
                st.error("שגיאה בהרצת הקוד שנוצר:")
                st.error(traceback.format_exc())
else:
    st.info("העלו קובץ Excel והקלידו שאלה כדי להתחיל.")
