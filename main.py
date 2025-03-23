import os
import time

scripts = [
    "src/data_ingestion.py",
    "src/data_transformation.py",
    "src/feature_engineering.py", 
    "src/model_building.py", 
    "src/model_evaluation.py"
]

all_success = True  # Flag to track script success

for script in scripts:
    exit_code = os.system(f"python {script}")
    if exit_code != 0:  # If any script fails
        print(f"❌ Error executing {script}. Stopping execution.")
        all_success = False
        break  # Stop running further scripts
    time.sleep(2)  # Wait 2 seconds before running the next script

if all_success:
    print("✅ All scripts executed successfully!")
else:
    print("❌ Some scripts failed. Not all scripts were executed.")



