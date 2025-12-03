# Fraud Detection System Using TCGM

A cost-sensitive fraud detection model built with the Time-Cost Gradient Machine (TCGM) algorithm. This system optimizes for financial impact by incorporating false positive and false negative costs directly into the model training process.

## 🎯 Overview

This project implements a fraud detection system that:
- Uses TCGM, a specialized gradient boosting algorithm for cost-sensitive classification
- Optimizes for Expected Monetary Loss (EML) rather than traditional accuracy metrics
- Incorporates business costs: $50 for false positives (investigation costs) and $200 for false negatives (fraud losses)
- Provides optimal decision thresholds based on financial impact

## 📊 Key Features

- **Cost-Sensitive Learning**: Direct incorporation of business costs into model training
- **Financial Metrics**: Evaluation using Expected Monetary Loss and cost-adjusted performance
- **Optimal Thresholding**: Automatic calculation of the best decision threshold for fraud classification
- **High Performance**: Achieves 98.3% AUC with low Brier score (0.0016)
- **Interpretable Results**: Loss curve visualization for threshold selection

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd fraud-detection-tcgm
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
fraud-detection-tcgm/
├── Fraud_tcgm.ipynb          # Main training notebook
├── fraud_features_ready.csv   # Preprocessed features
├── tcgm_model.pkl            # Trained model (after running)
├── tcgm_scaler.pkl           # Feature scaler (after running)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Usage

### Training the Model

Run the Jupyter notebook:
```bash
jupyter notebook Fraud_tcgm.ipynb
```

The notebook performs:
1. Data loading and train/test splitting (80/20)
2. Feature scaling using StandardScaler
3. TCGM model training with cost parameters
4. Model evaluation with financial metrics
5. Optimal threshold calculation
6. Model serialization

### Model Parameters

```python
model = TimeCostGradientMachine(
    n_estimators=60,        # Number of boosting iterations
    learning_rate=0.1,      # Learning rate
    max_depth=4,            # Maximum tree depth
    min_samples_leaf=20,    # Minimum samples per leaf
    cost_fp=50.0,          # False positive cost ($)
    cost_fn=200.0          # False negative cost ($)
)
```

### Making Predictions

```python
import joblib

# Load trained model and scaler
model = joblib.load("tcgm_model.pkl")
scaler = joblib.load("tcgm_scaler.pkl")

# Prepare your data
X_new = scaler.transform(your_features)

# Get fraud probabilities
fraud_probs = model.predict_proba(X_new)[:, 1]

# Apply optimal threshold
predictions = (fraud_probs >= optimal_threshold).astype(int)
```

## 📈 Model Performance

**Evaluation Metrics:**
- **AUC**: 0.983 (98.3%)
- **Brier Score**: 0.0016 (lower is better)
- **Expected Loss**: 0.41 per transaction
- **Optimal Threshold**: 0.94

**Cost Structure:**
- False Positive Cost: #50 (investigation cost)
- False Negative Cost: 200 (fraud loss)
- Loss Given Default (LGD): 60% (for EML calculation)

## 🔍 Key Components

### TCGM Algorithm
Time-Cost Gradient Machine is a gradient boosting variant that:
- Integrates misclassification costs during training
- Optimizes a cost-weighted loss function
- Produces calibrated probability estimates
- Handles imbalanced datasets effectively

### Expected Monetary Loss (EML)
Calculates financial impact considering:
- Transaction exposure amounts
- Probability of fraud
- Loss given default rate
- Investigation costs

### Optimal Threshold Selection
Uses loss curve analysis to find the threshold that minimizes expected financial loss across different classification boundaries.

## 📊 Metrics Explained

**evaluate_financial_performance()**: Returns AUC, Brier score, and expected loss based on cost parameters.

**compute_expected_monetary_loss()**: Computes the financial impact across different decision thresholds, returning:
- Best threshold for minimum loss
- Minimum expected loss
- Complete loss curve for visualization

## 🔧 Customization

### Adjusting Costs
Modify cost parameters based on your business context:
```python
cost_fp = 100.0  # Your false positive cost
cost_fn = 500.0  # Your false negative cost
lgd = 0.7        # Your loss given default rate
```

### Feature Engineering
The model expects preprocessed features in `fraud_features_ready.csv`. Ensure your feature engineering includes relevant fraud indicators.

## 📝 Data Requirements

Input data should include:
- Engineered fraud detection features
- Binary target variable (`isFraud`: 0 or 1)
- Transaction amount column for EML calculation
- Proper handling of missing values and encoding

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Hyperparameter tuning with RandomizedSearchCV
- Additional feature engineering
- Real-time prediction API
- Model monitoring and drift detection
- SMOTE or other sampling techniques for imbalance

## 📚 References

- [TCGM Library Documentation](https://pypi.org/project/tcgm/)


## 👤 Author

Chidiebere V. Christopher
- GitHub: 93Chidiebere
- Email: vchidiebere.vc@gmail.com

## 🙏 Acknowledgments

- Myself, TCGM library developer
- Scikit-learn community
- Imbalanced-learn contributors