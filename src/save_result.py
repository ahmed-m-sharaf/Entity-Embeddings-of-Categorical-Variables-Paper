import json
import os

def save_results_json(
    results: list[dict], ensemble_mape: float, file_path: str = "results.json"
):
    """
    Save training/testing results and ensemble MAPE to a JSON file.
    """
    data_to_save = {"models": results, "ensemble_mape": ensemble_mape}
    os.makedirs('results', exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data_to_save, f, indent=4)
    print(f"Results saved to '{file_path}'")
