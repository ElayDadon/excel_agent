# app.py — Streamlit + Gemini (new SDK) Excel Q&A in Hebrew
# ---------------------------------------------------------
# Requirements (requirements.txt):
# streamlit
# pandas
# openpyxl
# plotly
# google-genai>=1.30.0
#
# Secrets (Streamlit Community Cloud):
# GEMINI_API_KEY = "your_api_key"

import os
import re
import sys
import json
import math
import textwrap
import traceback
import importlib
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -------------------------- Streamlit page setup -----------------------------

st.set_page_config(
    page_title="Excel Q&A (Hebrew) — Gemini",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Excel Q&A (Hebrew) — Gemini 2.5 Flash")
st.caption("העלו קובץ אקסל, כתבו שאלה בעברית — והמודל ייצור קוד פייתון (pandas/plotly) שמריץ ומציג תשובה.")


# ------------------------------ Utilities -----------------------------------

def get_api_key() -> Optional[str]:
    """Get Gemini API key from Streamlit secrets or environment."""
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not key:
        key = os.environ.get("GEMINI_API_KEY")
    return key


def diagnose_google_shadowing() -> Optional[str]:
    """
    Warn only if a local 'google' dir or 'google.py' in the repo could shadow the SDK.
    Ignore the normal site-packages namespace package.
    """
    root = os.path.abspath(os.getcwd())
    hints = []
    conflict = False

    local_dir = os.path.join(root, "google")
    local_py  = os.path.join(root, "google.py")
    if os.path.isdir(local_dir):
        conflict = True
        hints.append(f"נמצאה תיקייה מקומית בשם 'google/': {local_dir}")
    if os.path.isfile(local_py):
        conflict = True
        hints.append(f"נמצא קובץ מקומי בשם 'google.py': {local_py}")

    mod = sys.modules.get("google")
    if mod is not None:
        paths = []
        pkg_path = getattr(mod, "__path__", None)
        if pkg_path:
            paths.extend(list(pkg_path))
        file = getattr(mod, "__file__", None)
        if file:
            paths.append(file)
        local_paths = [p for p in paths if p and os.path.abspath(p).startswith(root + os.sep)]
        if local_paths:
            conflict = True
            hints.append("מודול 'google' נטען ממסלולים בתוך הרפו: " + ", ".join(local_paths))

    return "\n".join(hints) if conflict else None


@st.cache_resource(show_spinner=False)
def get_client():
    """
    Lazily import google-genai here (NOT at module import time).
    This avoids failing before we can warn about shadowing and lets us show clear errors.
    """
    api_key = get_api_key()
    if not api_key:
        st.error("❌ לא נמצא מפתח API של Gemini. הוסף GEMINI_API_KEY ב-Secrets של Streamlit או כמשתנה סביבה.")
        st.stop()

    try:
        # from google import genai
        genai = importlib.import_module("google.genai")
    except Exception as e:
        diag = diagnose_google_shadowing()
        msg = [
            "❌ ImportError בעת טעינת 'google.genai' (SDK החדש).",
            "ודאו שאין תיקייה/קובץ בשם 'google' בתוך הרפו, ושב- requirements.txt מופיע רק google-genai (לא google-generativeai).",
        ]
        if diag:
            msg.append("\nאבחון:\n" + diag)
        msg.append(f"\nשגיאה מקורית: {type(e).__name__}: {e}")
        st.error("\n".join(msg))
        st.stop()

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"❌ יצירת Client נכשלה: {e}")
        st.stop()

    try:
        genai_types = importlib.import_module("google.genai.types")
    except Exception as e:
        st.error(f"❌ טעינת google.genai.types נכשלה: {e}")
        st.stop()

    return client, genai_types


def df_schema_and_sample(df: pd.DataFrame, max_rows: int = 5, max_len: int = 120) -> Dict[str, Any]:
    """Build a compact schema + sample to share with the model (no full data)."""
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample_df = df.head(max_rows).copy()
    for col in sample_df.columns:
        if sample_df[col].dtype == object:
            sample_df[col] = sample_df[col].astype(str).str.slice(0, max_len)
    return {"schema": schema, "sample_rows": sample_df.to_dict(orient="records")}


def extract_code_from_markdown(md: str) -> Optional[str]:
    """Extract first Python code block from markdown (```python ...``` or ``` ... ```)."""
    fence = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    m = fence.search(md or "")
    return m.group(1).strip() if m else None


def make_instruction_prompt() -> str:
    """System-style instruction for code generation (strongly constrained to avoid errors)."""
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

    STRICT RULES:
    - **Do NOT write any import lines.** Assume `pd` (pandas), `px` (plotly.express), and `go` (plotly.graph_objects)
      are already available.
    - Use ONLY pandas and plotly through those names.
    - Do NOT read or write files. Do NOT access network, OS, env, or time.
    - Prefer vectorized pandas ops (groupby, filters, dates) and simple Plotly figures.
    - Validate required columns. If missing, return a short Hebrew message explaining what's missing
      and suggest a reasonable alternative based on existing columns (return a string).
    - Keep charts readable: title, labels, sensible sorting/aggregation.
    - Keep the function small and deterministic. Avoid while/for loops over rows when possible.

    OUTPUT:
    - Return only the code of `solve` with no explanations.
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


def validate_code_safety(code: str) -> Optional[str]:
    """Return a reason string if code contains disallowed tokens; otherwise None."""
    # Disallow potentially dangerous modules/functions/attributes
    forbidden_patterns = [
        r"\b__import__\b",
        r"\bimportlib\b",
        r"\bos\b",
        r"\bsubprocess\b",
        r"\bsys\.\b",
        r"\bsocket\b",
        r"\brequests\b",
        r"\bopen\s*\(",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bcompile\s*\(",
        r"\bglobals\s*\(",
        r"\blocals\s*\(",
        r"\bgetattr\s*\(",
        r"\bsetattr\s*\(",
        r"\bdelattr\s*\(",
        r"\b__dict__\b",
    ]
    for pat in forbidden_patterns:
        if re.search(pat, code):
            return f"קוד שנוצר מכיל דפוס אסור: {pat}"
    if len(code) > 100_000:
        return "קוד שנוצר ארוך מדי."
    return None


def _strip_top_level_imports(code: str) -> str:
    """Remove top-level import lines; rely on host-provided pd/px/go."""
    pattern = re.compile(r'^\s*(?:import\s+[^\n]+|from\s+[^\n]+import\s+[^\n]+)\s*$', re.MULTILINE)
    return re.sub(pattern, '', code).strip()


def generate_solver_code(question: str, summary: Dict[str, Any], model_name: str, temperature: float) -> str:
    """Call Gemini (new SDK) to generate the solve(df, question) code."""
    (client, genai_types) = get_client()

    system_msg = make_instruction_prompt()
    user_msg = build_user_prompt(question, summary)

    resp = client.models.generate_content(
        model=model_name,
        contents=[system_msg, user_msg],
        config=genai_types.GenerateContentConfig(temperature=temperature),
    )

    text = (resp.text or "").strip()
    code = extract_code_from_markdown(text) or text
    if "def solve(" not in code:
        raise ValueError("התוכן שנוצר לא כולל פונקציה solve(df, question) תקינה.")
    # sanitize
    code = _strip_top_level_imports(code)
    reason = validate_code_safety(code)
    if reason:
        raise ValueError(f"הקוד שנוצר לא עבר בדיקת בטיחות: {reason}")
    return code


def safe_exec_solve(code: str) -> Tuple[Any, List[str]]:
    """Execute the generated code in a constrained namespace and return (solve, log)."""
    # capture prints
    exec_log: List[str] = []

    def _safe_print(*args, **kwargs):
        try:
            msg = " ".join(str(a) for a in args)
        except Exception:
            msg = "<unprintable>"
        exec_log.append(msg)

    # Minimal safe builtins + basic types for annotations
    safe_builtins = {
        "len": len, "range": range, "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "sorted": sorted, "enumerate": enumerate, "zip": zip, "any": any, "all": all, "map": map, "filter": filter,
        # common types
        "str": str, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        # allow print -> goes to exec_log
        "print": _safe_print,
    }

    allowed_globals = {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
        "pd": pd, "px": px, "go": go,
        "math": math, "datetime": datetime,
    }
    local_ns: Dict[str, Any] = {}

    exec(code, allowed_globals, local_ns)  # noqa: S102 (controlled)
    solve = local_ns.get("solve") or allowed_globals.get("solve")
    if not callable(solve):
        raise ValueError("לא נמצאה פונקציה בשם 'solve' לאחר הרצת הקוד.")
    return solve, exec_log


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
            st.plotly_chart(payload, use_container_width=True); return
        if isinstance(payload, pd.DataFrame):
            st.dataframe(payload, use_container_width=True); return
        if isinstance(payload, list):
            for item in payload:
                if hasattr(item, "to_plotly_json"):
                    st.plotly_chart(item, use_container_width=True)
                elif isinstance(item, pd.DataFrame):
                    st.dataframe(item, use_container_width=True)
                else:
                    st.write(item)
            return
        st.write(payload); return

    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True); return

    if hasattr(result, "to_plotly_json"):
        st.plotly_chart(result, use_container_width=True); return

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

diag = diagnose_google_shadowing()
if diag:
    st.warning("אזהרה: ייתכן שקיימת התנגשות שמות עם 'google'.\n" + diag, icon="⚠️")

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
        help="מומלץ: gemini-2.5-flash-001",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    show_code = st.toggle("להציג את הקוד שנוצר", value=False)
    show_log = st.toggle("להציג יומן הדפסות", value=False)

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
                solve, exec_log = safe_exec_solve(code)
                out = solve(df, question.strip())
                render_result(out)
                if show_log and exec_log:
                    st.markdown("#### יומן הדפסות מהקוד")
                    for line in exec_log:
                        st.text(line)
            except Exception:
                st.error("שגיאה בהרצת הקוד שנוצר:")
                st.error(traceback.format_exc())
else:
    st.info("העלו קובץ Excel והקלידו שאלה כדי להתחיל.")
