import os
import subprocess
from openai import OpenAI

client = OpenAI(api_key="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")

SCRIPT_FILE = "emulator_script.py"
OUTPUT_DIR = "emulator_screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

user_instructions = "In the emulator window, open edge browser,make it window mode click on a new tab and search for coconuts"

# --- Ask GPT‑4o‑mini for automation code ---
prompt = f"""
Write a Python script that:
1. Uses pyautogui to interact with an emulator window to follow these instructions: {user_instructions}.
2. Takes a screenshot after each action but just before executing them.
3. Saves screenshots into the folder '{OUTPUT_DIR}' as step1.png, step2.png, etc.
4. Uses short delays between actions.
5. Does not use input().
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an assistant that writes safe Python automation scripts to control emulator windows. Just give bare code with no explanations or annotations."},
        {"role": "user", "content": prompt}
    ]
)

# --- Strip Markdown fences if present ---
generated_code = response.choices[0].message.content.strip()
if generated_code.startswith("```"):
    generated_code = generated_code.split("\n", 1)[1]  # remove first ```
if generated_code.endswith("```"):
    generated_code = generated_code.rsplit("\n", 1)[0]  # remove last ```

# --- Save to .py file ---
with open(SCRIPT_FILE, "w", encoding="utf-8") as f:
    f.write(generated_code)

print(f"[INFO] Running emulator automation script...")
subprocess.run(["python", SCRIPT_FILE], check=True)
print(f"[INFO] Screenshots saved in '{OUTPUT_DIR}'")
