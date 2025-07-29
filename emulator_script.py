import pyautogui
import time
import os

# Ensure the screenshot directory exists
os.makedirs('emulator_screenshots', exist_ok=True)

# Step 1: Take a screenshot and open Edge
pyautogui.screenshot('emulator_screenshots/step1.png')
pyautogui.hotkey('command', 'space')
time.sleep(1)
pyautogui.write('Microsoft Edge')
time.sleep(1)
pyautogui.press('return')
time.sleep(5)

# Step 2: Take a screenshot and make Edge windowed
pyautogui.screenshot('emulator_screenshots/step2.png')
pyautogui.hotkey('command', 'm')  # Minimize to get the windowed mode, this command may vary
time.sleep(1)
pyautogui.hotkey('command', 'm')  # Restore
   
# Step 3: Take a screenshot and click new tab
pyautogui.screenshot('emulator_screenshots/step3.png')
pyautogui.hotkey('command', 't')
time.sleep(1)

# Step 4: Take a screenshot and search for coconuts
pyautogui.screenshot('emulator_screenshots/step4.png')
pyautogui.write('coconuts')
time.sleep(1)
pyautogui.press('return')