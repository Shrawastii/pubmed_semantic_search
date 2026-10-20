import os, datetime
import pandas as pd

RESULTS_FILE = "results/results.csv"

def log_result(system, **metrics):
    os.makedirs("results", exist_ok=True)
    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
           "system": system, **metrics}
    new = pd.DataFrame([row])
    if os.path.exists(RESULTS_FILE):
        new = pd.concat([pd.read_csv(RESULTS_FILE), new], ignore_index=True)
    new.to_csv(RESULTS_FILE, index=False)