# MLForge Credit Card Fraud Detection

A production-ready MLOps project for credit card fraud detection using PyTorch, DVC, and MLflow.

## 🎯 Project Overview

This project implements a neural network-based fraud detection system for credit card transactions. The model is trained on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) and achieves high recall for identifying fraudulent transactions while managing the severe class imbalance inherent in fraud detection problems.

**Key Features:**
- ✅ **Modular codebase** with separation of concerns
- ✅ **DVC pipeline** for reproducible data processing and training
- ✅ **MLflow tracking** for experiment management
- ✅ **Comprehensive testing** (unit + integration tests)
- ✅ **CI/CD pipeline** with GitHub Actions
- ✅ **Class-weighted loss** to handle imbalanced data

## 📊 Dataset

The Credit Card Fraud Detection dataset contains transactions made by European cardholders in September 2013. 

**Dataset Statistics:**
- **Total transactions:** 284,807
- **Fraudulent transactions:** 492 (0.172%)
- **Features:** 30 (28 PCA-transformed features + Time + Amount)
- **Target:** Binary classification (0 = Legitimate, 1 = Fraud)

**Class Imbalance:** The dataset is highly imbalanced with fraudulent transactions representing only 0.172% of all transactions. This project addresses this challenge using:
- StandardScaler normalization for Time and Amount features
- Stratified train/test split to preserve class distribution
- Class-weighted BCEWithLogitsLoss during training

## 🏗️ Model Architecture

**FraudNet** - A 4-layer feedforward neural network:

```
Input (30 features)
    ↓
Linear(30 → 256) + ReLU
    ↓
Linear(256 → 256) + ReLU
    ↓
Linear(256 → 256) + ReLU
    ↓
Linear(256 → 1) [logits]
    ↓
Sigmoid → Binary prediction
```

**Training Configuration:**
- **Loss Function:** BCEWithLogitsLoss with positive class weighting
- **Optimizer:** Adam (lr=0.005)
- **Epochs:** 100 (configurable via params.yaml)
- **Batch Processing:** Full-batch training

## 📁 Project Structure

```
mlcredit/
├── data/
│   ├── raw/                    # Raw dataset (tracked by DVC)
│   │   └── creditcard.csv
│   ├── interim/                # Cleaned data
│   │   └── clean.csv
│   └── processed/              # Preprocessed numpy arrays
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
├── models/
│   └── model.pt                # Trained PyTorch model
├── reports/
│   ├── metrics.json            # Evaluation metrics
│   └── figures/
│       └── confusion_matrix.png
├── src/
│   ├── model.py                # FraudNet architecture
│   ├── load_data.py            # Data loading stage
│   ├── preprocess.py           # Preprocessing stage
│   ├── train.py                # Training stage
│   └── evaluate.py             # Evaluation stage
├── tests/
│   ├── test_data_pipeline.py   # Data pipeline tests
│   ├── test_training.py        # Training tests
│   └── test_model_artifact.py  # Artifact validation tests
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # Configuration parameters
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- (Optional) DVC remote storage for data versioning

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd mlcredit
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Place the dataset:**
```bash
# Download creditcard.csv and place it in data/raw/
mkdir -p data/raw
# Copy your creditcard.csv to data/raw/
```

4. **Initialize DVC (optional):**
```bash
dvc init
dvc add data/raw/creditcard.csv
```

5. **Configure DVC Remote (if using Google Drive):**
The project is configured to use Google Drive. You may need to authenticate:
```bash
# This will trigger an authentication flow in your browser
dvc pull
```
If that doesn't work, you might need to configure your own remote:
```bash
dvc remote add -d storage gdrive://<YOUR_FOLDER_ID>
```

## 🔄 Running the Pipeline

### Option 1: Run Complete DVC Pipeline

Execute all stages (load → preprocess → train → evaluate):

```bash
dvc repro
```

This will:
1. Load and validate raw data
2. Apply feature scaling and train/test split
3. Train the FraudNet model with MLflow tracking
4. Evaluate the model and generate metrics + visualizations

### Option 2: Run Individual Stages

**Data Loading:**
```bash
python src/load_data.py
```

**Preprocessing:**
```bash
python src/preprocess.py
```

**Training:**
```bash
python src/train.py
```

**Evaluation:**
```bash
python src/evaluate.py
```

## 📈 MLflow Experiment Tracking

### View Experiments

Start the MLflow UI:
```bash
mlflow ui
```

Then open your browser to `http://localhost:5000`

### What's Tracked

**Parameters:**
- Model architecture (input_features, hidden_units, output_features)
- Training config (epochs, learning_rate, random_seed)
- Data split (test_size)
- Class weights (pos_weight)

**Metrics (per epoch):**
- train_loss, train_acc
- test_loss, test_acc

**Artifacts:**
- model.pt (trained model state dict)
- metrics.json (evaluation metrics)
- confusion_matrix.png (visualization)

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suites

**Data Pipeline Tests:**
```bash
pytest tests/test_data_pipeline.py -v
```

**Training Tests:**
```bash
pytest tests/test_training.py -v
```

**Model Artifact Tests:**
```bash
pytest tests/test_model_artifact.py -v
```

### Test Coverage

The test suite includes:
- ✅ Data loading and validation
- ✅ Preprocessing and feature scaling
- ✅ Train/test split verification
- ✅ Model training with epoch override
- ✅ Model architecture validation
- ✅ Evaluation metrics validation
- ✅ Artifact existence and format checks

## ⚙️ Configuration

All configuration is centralized in `params.yaml`:

```yaml
# Data paths
data:
  raw: data/raw/creditcard.csv
  interim: data/interim/clean.csv
  processed: { ... }

# Model architecture
model:
  input_features: 30
  hidden_units: 256
  output_features: 1

# Training configuration
training:
  epochs: 100
  learning_rate: 0.005
  random_seed: 42

# MLflow configuration
mlflow:
  experiment_name: credit-fraud-detection
  tracking_uri: ./mlruns
```

### Environment Variables

**N_EPOCHS_OVERRIDE:** Override the number of training epochs (useful for CI/CD)

```bash
N_EPOCHS_OVERRIDE=2 python src/train.py
```

## 🔄 CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:

1. ✅ Sets up Python 3.9 environment
2. ✅ Installs dependencies with caching
3. ✅ Pulls DVC data (with graceful fallback)
4. ✅ Runs pytest test suite
5. ✅ Executes DVC pipeline with epoch override (N_EPOCHS_OVERRIDE=2)
6. ✅ Uploads model and metrics as artifacts

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`

**Artifacts Uploaded:**
- `trained-model` (models/model.pt)
- `evaluation-metrics` (reports/)
- `mlflow-runs` (mlruns/)

## 📊 Evaluation Metrics

After running the evaluation stage, metrics are saved to `reports/metrics.json`:

```json
{
  "precision": 0.XXXX,
  "recall": 0.XXXX,
  "f1_score": 0.XXXX,
  "accuracy": 0.XXXX
}
```

**Confusion Matrix:** A visualization is saved to `reports/figures/confusion_matrix.png` showing:
- True Negatives (legitimate transactions correctly identified)
- False Positives (legitimate transactions flagged as fraud)
- False Negatives (fraudulent transactions missed)
- True Positives (fraudulent transactions correctly identified)

## 🔧 Development

### Adding New Features

1. Update `params.yaml` with new configuration
2. Modify relevant source files in `src/`
3. Update `dvc.yaml` if pipeline stages change
4. Add tests in `tests/`
5. Run tests: `pytest tests/ -v`
6. Run pipeline: `dvc repro`

### Modifying Model Architecture

Edit `src/model.py` and update the `FraudNet` class. Remember to:
- Update `params.yaml` if new hyperparameters are added
- Retrain the model: `dvc repro train`
- Re-evaluate: `dvc repro evaluate`

## 📝 Original Implementation

The original single-file implementation is preserved as `creditfraud.py` for reference.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Original research: Andrea Dal Pozzolo et al.

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built with ❤️ using PyTorch, DVC, and MLflow**
