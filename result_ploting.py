from src.visualization import plot_losses, plot_mape
import sys

if len(sys.argv) > 2:
    file_path = sys.argv[1]
    name = sys.argv[2]
    plot_losses(file_path, f'{name}_loss')
    plot_mape(file_path, f'{name}_mape')
else :
    print("Must Set Path json file results and name of figure")