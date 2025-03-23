import os
import time

scripts = [
    "src/data_ingestion.py",
    "src/data_transformation.py",
    "src/feature_engineering.py"
]

for script in scripts:
    exit_code = os.system(f"python {script}")
    if exit_code != 0:  # Check if the script failed
        print(f"❌ Error executing {script}, stopping execution.")
        break
    time.sleep(2)  # Wait 2 seconds before running the next script

print("✅ All scripts executed successfully!")



