# Entity Embedding for Rossmann Store Sales Prediction

## About the Project

This project implements Entity Embedding neural networks for sales forecasting using the Rossmann Store Sales dataset from Kaggle. Entity embeddings are learned representations of categorical variables that capture their intrinsic properties and relationships, making them effective for structured data modeling.

The project explores various model configurations including:
- **Original Model**: Random-based and time-based data splitting
- **Ablation Studies**: 
  - Different embedding dimensions
  - Batch normalization effects
  - Learning rate variations

The model uses a deep neural network with embedding layers for categorical features (Store, DayOfWeek, Date components, Promo, State) followed by fully connected layers to predict store sales.

## Project Structure

```
Entity Embedding/
│
├── data/                          # Dataset directory
│   ├── train.csv                  # Training data (download from Kaggle)
│   └── store_states.csv          # Store state mapping
│
├── src/                           # Source code modules
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration parameters
│   ├── data_module.py            # Data loading utilities
│   ├── embedding_visualization.py # t-SNE visualization of embeddings
│   ├── metrics.py                # Evaluation metrics (MAPE)
│   ├── model.py                  # RossmannEmbeddingModel definition
│   ├── preprocessor.py           # Data preprocessing functions
│   ├── rossmann_store.py         # Rossmann dataset handler
│   ├── save_result.py            # Results saving utilities
│   ├── trainer.py                # Model training logic
│   └── visualization.py          # Loss and MAPE plotting
│
├── models/                        # Saved model checkpoints
│   ├── Original_Model_Random_Based/
│   ├── Original_Model_Time_Based/
│   ├── Ablation_With_Batch_Normalization/
│   ├── Ablation_With_Embedding_Dim/
│   └── Ablation_With_Learning_Rate/
│
├── images/                        # Generated visualizations
│   ├── Original_Model_Random_Based/
│   ├── Original_Model_Time_Based/
│   └── [Ablation experiments visualizations]
│
├── results/                       # Experiment results (JSON)
│   ├── Original_Model_Random_Based_result.json
│   ├── Original_Model_Time_Based_result.json
│   └── [Ablation experiment results]
│
├── main.py                        # Main training script
├── setup.py                       # Experiment configurations
├── utilis.py                      # Training utilities and helper functions
├── result_ploting.py              # Result visualization script
├── target_max.json               # Target normalization metadata
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Dataset

### Download from Kaggle

This project uses the **Rossmann Store Sales** dataset from Kaggle:

**Competition URL**: https://www.kaggle.com/competitions/rossmann-store-sales

#### Steps to Download:

1. **Create a Kaggle account** (if you don't have one): https://www.kaggle.com/

2. **Install Kaggle API** (optional but recommended):
   ```bash
   pip install kaggle
   ```

3. **Set up Kaggle API credentials**:
   - Go to your Kaggle account settings: https://www.kaggle.com/[username]/account
   - Scroll to "API" section and click "Create New API Token"
   - This will download `kaggle.json`
   - Place it in:
     - **Windows**: `C:\Users\<Username>\.kaggle\kaggle.json`
     - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - Set permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

4. **Download the dataset using Kaggle API**:
   ```bash
   kaggle competitions download -c rossmann-store-sales
   ```

5. **Extract the files**:
   ```bash
   # Windows PowerShell
   Expand-Archive rossmann-store-sales.zip -DestinationPath data/
   
   # Or manually extract train.csv to the data/ folder
   ```

6. **Alternative - Manual Download**:
   - Visit: https://www.kaggle.com/competitions/rossmann-store-sales/data
   - Click "Download All" or download individual files
   - Extract `train.csv` to the `data/` directory

**Required files:**
- `train.csv` → Place in `data/train.csv`
- Create `store_states.csv` manually or ensure it exists with store-to-state mappings

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation Steps

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Data Preparation

Once you have the dataset downloaded, the data preprocessing is handled automatically by the training script. The preprocessing includes:

1. **Loading and merging data**: Combines `train.csv` with `store_states.csv`
2. **Filtering**: Removes closed stores and days with zero sales
3. **Feature engineering**: Extracts day, month, year from dates
4. **Target transformation**: Applies log transformation and normalization
5. **Categorical encoding**: Encodes categorical features using Ordinal Encoding
6. **Data splitting**: Creates train/test splits (random or time-based)

No manual data preparation is required - just ensure the CSV files are in the `data/` directory.

## Training the Model

### Basic Trainin

To train all configured experiments:

```bash
python main.py
```

This will:
- Train 5 ensemble models for each experiment configuration
- Save trained models to `models/` directory
- Generate embedding visualizations in `images/` directory
- Save results to `results/` directory as JSON files
- Print ensemble MAPE scores for each experiment

### Configuration

Modify training parameters in [src/config.py](src/config.py):

```python
# Data Configuration
DATA_PATH = "data/train.csv"
STORE_STATE = "data/store_states.csv"
SPLIT_TYPE = "random"  # or "time"
TEST_SIZE = 0.1

# Model Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
EMB_DIMS = {
    "Store": 10,
    "DayOfWeek": 6,
    "day": 10,
    "month": 6,
    "year": 2,
    "Promo": 1,
    "State": 6,
}
RANDOM_STATE = 42
```

### Experiment Configurations

The [setup.py](setup.py) file defines multiple experiments:
- `Original_Model_Random_Based`: Random shuffle training
- `Original_Model_Time_Based`: Time-series split (no shuffle)
- `Ablation_With_Embedding_Dim`: Larger embedding dimensions
- `Ablation_With_Batch_Normalization`: With batch normalization layers
- `Ablation_With_Learning_Rate_Increasing/Decreasing`: Different learning rates

## Model Evaluation

### Automatic Evaluation

Evaluation happens automatically during training. Metrics include:
- **MAPE** (Mean Absolute Percentage Error): Primary evaluation metric
- **Training Loss**: MSE loss on training set
- **Validation Loss**: MSE loss on test set

### View Results

After training completes:

1. **Check console output** for ensemble MAPE scores

2. **View JSON results** in `results/` directory:
   ```bash
   # Windows PowerShell
   Get-Content results/Original_Model_Random_Based_result.json
   
   # Or open in any text editor
   ```

3. **Generate visualizations**:
   ```bash
   python result_ploting.py
   ```

4. **Explore embedding visualizations**:
   - Check `images/` directory for t-SNE plots of learned embeddings
   - Each experiment has separate folders with visualizations per model

### Results Structure

JSON result files contain:
```json
{
    "models": [
        {
            "model": 1,
            "train_mape": 0.1234,
            "test_mape": 0.1456,
            "history": {
                "train_losses": [...],
                "val_losses": [...]
            }
        },
        ...
    ],
    "ensemble_mape": 0.1401
}
```

## Model Architecture

The `RossmannEmbeddingModel` consists of:
1. **Embedding Layers**: Separate embeddings for each categorical feature
2. **Fully Connected Layers**: 
   - FC1: [total_embedding_dim] → 1000
   - FC2: 1000 → 500
   - Output: 500 → 1
3. **Activation**: ReLU
4. **Optional**: Batch normalization layers
5. **Output**: Sigmoid activation (optional)

## Dependencies

See [requirements.txt](requirements.txt) for full list:
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Preprocessing and t-SNE visualization
- **Matplotlib**: Plotting and visualization

## Citation

If you use this code or the Rossmann dataset, please cite:

```
Rossmann Store Sales Competition
Kaggle, 2015
https://www.kaggle.com/competitions/rossmann-store-sales
```

## License

This project is for educational and research purposes.

## Contact & Support

For issues or questions, please open an issue in the repository or refer to the Kaggle competition discussion forums.
