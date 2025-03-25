# Diabetes Prediction Pipeline with Interactive UI

![UI Demo](https://github.com/Kotka334/Detecting-Diabetes/blob/main/ui.jpg?raw=true)

This project implements a machine learning pipeline for predicting diabetes using a cleaned clinical dataset. It includes both model training logic and a user-friendly Gradio-based interface for interactive prediction and data visualization.

## 📁 Project Structure

- **`model_training.py`** – Builds and trains various classification models (Logistic Regression, Random Forest, XGBoost, LightGBM) on a structured health dataset. It includes preprocessing pipelines, cross-validation, and hyperparameter tuning using GridSearchCV and TPOT.
- **`plot.py`** – Implements an interactive web interface using Gradio. Users can input patient data and get real-time predictions along with visualizations of key health indicators.

## 🔍 Features

- Preprocessing with imputation, scaling, and encoding
- Model selection with evaluation metrics (ROC AUC, classification report)
- Export of the best model as `best_diabetes_model.pkl`
- Gradio UI for:
  - Entering patient input
  - Viewing diabetes prediction (Diabetic / Not Diabetic)
  - Plotting clinical features using Seaborn and Matplotlib

## 🛠 Installation

1. Clone this repository.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

Suggested `requirements.txt` contents:
```txt
pandas
numpy
scikit-learn
xgboost
lightgbm
tpot
gradio
matplotlib
seaborn
joblib
```

3. Ensure the file `Cleaned_Diabetes_Dataset.csv` is in the project root directory.

## 🚀 Usage

### Train Models
To train and save the best model:

```bash
python model_training.py
```

This will output `best_diabetes_model.pkl`.

## 📈 Model Performance

| Model               | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9189   | 0.7698    | 0.7252 | 0.7453   | 0.8811  |
| Random Forest       | 0.9730   | 0.9682    | 0.9195 | 0.9426   | 0.9995  |
| XGBoost             | 0.9865   | 0.9751    | 0.9751 | 0.9751   | 0.9996  |
| LightGBM            | 0.9932   | 0.9973    | 0.9778 | 0.9872   | 0.9998  |
| Voting Ensemble     | 0.9932   | 0.9792    | 0.9973 | 0.9879   | 0.9996  |
| TPOT                | 0.9730   | 0.9682    | 0.9195 | 0.9426   | 0.9992  |

### Launch Gradio UI
Once the model is trained, start the UI:

```bash
python plot.py
```

A browser window will open allowing you to interact with the model.

## 📊 Dataset

Ensure that `Cleaned_Diabetes_Dataset.csv` includes at least the following columns:

- `Gender`, `AGE`, `Urea`, `Cr`, `HbA1c`, `Chol`, `TG`, `HDL`, `LDL`, `VLDL`, `BMI`, `CLASS`

The target variable is `CLASS` (0: Not Diabetic, 1: Diabetic).

## 📌 Notes

- `model_training.py` supports both manual and automated model selection.
- UI uses the best saved model (`best_diabetes_model.pkl`) for inference.
- Designed for educational and demonstration purposes.

## 📧 Contact

For questions or suggestions, feel free to open an issue or reach out.
