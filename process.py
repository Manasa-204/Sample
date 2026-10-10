import os
import time
import subprocess
import threading
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

# Save the Excel file after every 50 questions
SAVE_EVERY = 50

# Target maximum GPU temperature (C) - runs at full speed and throttles only near 83C
MAX_GPU_TEMP = 83
current_gpu_temp = 60

def _monitor_gpu_temp():
    global current_gpu_temp
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                creationflags=subprocess.CREATE_NO_WINDOW
            ).decode().strip()
            current_gpu_temp = int(out.splitlines()[0])
        except Exception:
            pass
        time.sleep(2.0)

threading.Thread(target=_monitor_gpu_temp, daemon=True).start()

# ============================================================
# LOAD EXCEL
# ============================================================

if os.path.exists(OUTPUT_FILE):
    print(f"Resuming from existing output file: {OUTPUT_FILE}...")
    df = pd.read_excel(OUTPUT_FILE, sheet_name=0)
else:
    print(f"Loading input file: {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE, sheet_name=0)

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

df["Qwen_Ollama_Answers"] = df["Qwen_Ollama_Answers"].astype(object)
df["Qwen_Status"] = df["Qwen_Status"].astype(object)
df["Qwen_Error"] = df["Qwen_Error"].astype(object)

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
                "temperature": 0,
                "num_gpu": 99
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
    # DYNAMIC THERMAL GUARD (KEEPS GPU <= 83C AT FULL SPEED)
    # --------------------------------------------------------
    if current_gpu_temp >= MAX_GPU_TEMP:
        time.sleep(0.35)
    elif current_gpu_temp >= MAX_GPU_TEMP - 2:
        time.sleep(0.15)
    else:
        time.sleep(0.05)

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    if (index + 1) % SAVE_EVERY == 0:

        print(f"\nSaving checkpoint at row {index + 1}...")

        temp_file = OUTPUT_FILE + ".tmp.xlsx"
        df.to_excel(
            temp_file,
            index=False,
            engine="openpyxl"
        )
        os.replace(temp_file, OUTPUT_FILE)

# ============================================================
# FINAL SAVE
# ============================================================

print("\nSaving final Excel file...")

temp_file = OUTPUT_FILE + ".tmp.xlsx"
df.to_excel(
    temp_file,
    index=False,
    engine="openpyxl"
)
os.replace(temp_file, OUTPUT_FILE)

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