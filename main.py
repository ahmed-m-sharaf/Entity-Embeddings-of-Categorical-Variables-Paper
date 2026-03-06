import torch
from utilis import *
from setup import *


if __name__ == "__main__":
    
    # Configuration 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for exp_name, config in all_setups.items():
        all_results, ensemble_mape = Experiment(**config, model_name=exp_name)
        print(f"Ensemble Final Result {exp_name} : {ensemble_mape}")
        save_results_json(
            all_results, ensemble_mape, file_path = os.path.join('results', f'{exp_name}_result.json')
        )
    