import pyautogui
import time
import os

# Create folder for screenshots
os.makedirs('emulator_screenshots', exist_ok=True)

# Step 1: Open Safari browser
pyautogui.screenshot('emulator_screenshots/step1.png')
pyautogui.hotkey('command', 'space')  # Open Spotlight
time.sleep(1)
pyautogui.typewrite('Safari')
time.sleep(1)
pyautogui.press('return')
time.sleep(3)

# Step 2: Make Safari window mode
pyautogui.screenshot('emulator_screenshots/step2.png')
pyautogui.hotkey('command', 'control', 'f')  # Toggle fullscreen
time.sleep(1)
pyautogui.hotkey('command', 't')  # New tab
time.sleep(1)

# Step 3: Search for coconuts
pyautogui.screenshot('emulator_screenshots/step3.png')
pyautogui.typewrite('coconuts')
time.sleep(1)
pyautogui.press('return')