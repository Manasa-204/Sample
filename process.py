import pandas as pd
import ollama
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "QA_SLMForge_LabourCodes_10k_Dataset.xlsx"
OUTPUT_FILE = "QA_SLMForge_LabourCodes_10k_Qwen_Ollama.xlsx"

MODEL = "qwen2.5:7b"

QUESTION_COLUMN = "Employee_Questions"

# Save the Excel file after every 100 questions
SAVE_EVERY = 100

# ============================================================
# LOAD EXCEL
# ============================================================

print("Loading Excel file...")

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=0
)

print(f"Loaded {len(df)} rows.")
print(f"Using column: {QUESTION_COLUMN}")

# ============================================================
# CREATE OUTPUT COLUMNS
# ============================================================

if "Qwen_Ollama_Answers" not in df.columns:
    df["Qwen_Ollama_Answers"] = ""

if "Qwen_Status" not in df.columns:
    df["Qwen_Status"] = ""

if "Qwen_Error" not in df.columns:
    df["Qwen_Error"] = ""

# ============================================================
# PROCESS QUESTIONS
# ============================================================

for index in tqdm(
    range(len(df)),
    desc="Processing questions"
):

    # --------------------------------------------------------
    # Skip rows that were already processed
    # --------------------------------------------------------

    existing_answer = df.at[index, "Qwen_Ollama_Answers"]

    if pd.notna(existing_answer) and str(existing_answer).strip():
        continue

    # --------------------------------------------------------
    # Get question
    # --------------------------------------------------------

    question = df.at[index, QUESTION_COLUMN]

    if pd.isna(question) or not str(question).strip():

        df.at[index, "Qwen_Status"] = "Skipped"
        df.at[index, "Qwen_Error"] = "Empty question"

        continue

    question = str(question).strip()

    # --------------------------------------------------------
    # Prompt Qwen
    # --------------------------------------------------------

    prompt = f"""
You are an assistant answering employee questions related to
Indian labour laws and employment policies.

Answer the following employee question accurately and directly.

Do not invent legal provisions, sections, percentages, dates,
benefits, penalties, or other facts.

If the information is insufficient to give a reliable answer,
state that clearly.

Employee Question:
{question}

Give only the answer to the employee's question.
"""

    # --------------------------------------------------------
    # SEND QUESTION TO OLLAMA
    # --------------------------------------------------------

    try:

        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        answer = response["message"]["content"].strip()

        df.at[index, "Qwen_Ollama_Answers"] = answer
        df.at[index, "Qwen_Status"] = "Success"
        df.at[index, "Qwen_Error"] = ""

    except Exception as e:

        df.at[index, "Qwen_Status"] = "Error"
        df.at[index, "Qwen_Error"] = str(e)

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    if (index + 1) % SAVE_EVERY == 0:

        print(f"\nSaving checkpoint at row {index + 1}...")

        df.to_excel(
            OUTPUT_FILE,
            index=False,
            engine="openpyxl"
        )

# ============================================================
# FINAL SAVE
# ============================================================

print("\nSaving final Excel file...")

df.to_excel(
    OUTPUT_FILE,
    index=False,
    engine="openpyxl"
)

# ============================================================
# SUMMARY
# ============================================================

success = (df["Qwen_Status"] == "Success").sum()
errors = (df["Qwen_Status"] == "Error").sum()
skipped = (df["Qwen_Status"] == "Skipped").sum()

print("\n========================================")
print("PROCESSING COMPLETE")
print("========================================")
print(f"Total rows : {len(df)}")
print(f"Successful : {success}")
print(f"Errors     : {errors}")
print(f"Skipped    : {skipped}")
print(f"Output     : {OUTPUT_FILE}")
print("========================================")