# agent_logic.py
from google import genai
from google.genai import types
import pandas as pd

def create_gemini_client(api_key):
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error creating Gemini client: {e}")
        return None

def get_response_from_gemini(client, dataframe, query, chat_history):
    if not client:
        return "Error: Gemini client is not initialized."
    
    # --- Using the correct and available model ---
    model = "gemini-2.0-flash"
    
    df_columns = dataframe.columns.tolist()
    
    # --- The new and improved prompt ---
    prompt_text = f'''
    You are a world-class, friendly, and conversational data analyst AI.
    Your main goal is to help a user understand their data by answering questions in natural Hebrew.
    You specialize in Python with pandas and Plotly.

    **CONTEXT:**
    - You have access to a pandas DataFrame named `df` with the following columns: {df_columns}.
    - The user's current query is: "{query}"
    - The recent conversation history is: {chat_history}

    **YOUR CORE PRINCIPLES:**
    1.  **Be Conversational and Proactive:** When first asked "what is in the file" (e.g., "מה יש בקובץ"), provide a comprehensive text-only overview. This overview should include:
        - A brief summary of the data's purpose.
        - The number of rows and columns (from `df.shape`).
        - A list of all column names.
        - A suggestion for next steps, like "I can provide a statistical summary, show the first few rows, or create a graph. What would you like to do?".
        - **Do not generate code for this initial overview.**
    2.  **Understand User Intent & Handle Typos:** Your top priority is to understand what the user *means*, not just what they typed.
        - **Crucial Example:** If the user asks "מה יש בקבוץ", you MUST interpret it as a typo for "מה יש בקובץ" (what is in the file) and provide the initial overview. Do NOT assume they mean "group".
    3.  **Use Conversational History:** Look at the `chat_history`. If you just provided a statistical summary and the user asks for it again, gently point it out and ask if they want something different.
    4.  **Handle Ambiguity:** If a query is vague like "תספק לי" (Provide me), use the immediate preceding context. If your last message offered a statistical summary, it's logical to assume that's what the user wants. Generate the code for `df.describe()`.

    **RESPONSE FORMATTING (Strictly follow this):**
    - To generate pandas code for data/analysis, wrap it in ```python\\n[CODE]\\n...```
    - To generate Plotly code for a graph, wrap it in ```python\\n[PLOT]\\n...```
    - For conversational text answers (like the initial overview), respond directly without any wrappers.
    '''
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])]
    try:
        response = client.models.generate_content(model=f"models/{model}", contents=contents)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"

def execute_code(dataframe, gemini_response):
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