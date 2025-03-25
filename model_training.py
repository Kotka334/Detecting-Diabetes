import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 导入各模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 导入 TPOT 用于自动化模型搜索（可选）
from tpot import TPOTClassifier

# --------------------------
# 1. 数据加载与预处理
# --------------------------
# 读取数据（确保 Cleaned_Diabetes_Dataset.csv 与此脚本在同一目录下）
data = pd.read_csv('Cleaned_Diabetes_Dataset.csv')

# 假设数据中包含如下列：Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI, CLASS
X = data.drop('CLASS', axis=1)
y = data['CLASS']

# 数值型特征（除 Gender 外）
numeric_features = [col for col in X.columns if col != 'Gender']
# 分类特征
categorical_features = ['Gender']

# 定义预处理管道：对数值特征填补缺失值并标准化，对分类特征做 one-hot 编码（如果二分类，则可以设置 drop='if_binary'）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(drop='if_binary'), categorical_features)
    ]
)

# 分层划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------
# 2. 分别使用四种模型进行参数优化
# --------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 定义各模型的参数搜索空间（这里仅列举了部分参数，可根据需要扩展）
param_grid_lr = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear']  # 使用 liblinear 可支持 l1 惩罚
}

param_grid_rf = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 5, 10],
    'clf__min_samples_split': [2, 5],
    'clf__max_features': ['sqrt', 'log2']
}

param_grid_xgb = {
    'clf__learning_rate': [0.01, 0.1],
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5, 7]
}

param_grid_lgb = {
    'clf__learning_rate': [0.01, 0.1],
    'clf__n_estimators': [100, 200],
    'clf__num_leaves': [31, 50]
}

# 构建各模型的流水线（流水线中先执行预处理，再调用分类器）
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])

pipeline_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])

pipeline_lgb = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LGBMClassifier(random_state=42))
])

# 利用 GridSearchCV 对每个模型进行超参数搜索
print("正在优化 Logistic Regression ...")
grid_lr = GridSearchCV(pipeline_lr, param_grid=param_grid_lr, cv=cv, scoring='roc_auc_ovo', n_jobs=-1, verbose=1)
grid_lr.fit(X_train, y_train)

print("正在优化 Random Forest ...")
grid_rf = GridSearchCV(pipeline_rf, param_grid=param_grid_rf, cv=cv, scoring='roc_auc_ovo', n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)

print("正在优化 XGBoost ...")
grid_xgb = GridSearchCV(pipeline_xgb, param_grid=param_grid_xgb, cv=cv, scoring='roc_auc_ovo', n_jobs=-1, verbose=1)
grid_xgb.fit(X_train, y_train)

print("正在优化 LightGBM ...")
grid_lgb = GridSearchCV(pipeline_lgb, param_grid=param_grid_lgb, cv=cv, scoring='roc_auc_ovo', n_jobs=-1, verbose=1)
grid_lgb.fit(X_train, y_train)

# --------------------------
# 3. 模型评估
# --------------------------
def print_evaluation(model, X_test, y_test, model_name):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    print(f"\n=== {model_name} 评估结果 ===")
    print(classification_report(y_test, pred, digits=4))
    auc = roc_auc_score(y_test, proba, multi_class='ovo')
    print(f"{model_name} AUC: {auc:.4f}")
    return auc

auc_lr = print_evaluation(grid_lr, X_test, y_test, "Logistic Regression")
auc_rf = print_evaluation(grid_rf, X_test, y_test, "Random Forest")
auc_xgb = print_evaluation(grid_xgb, X_test, y_test, "XGBoost")
auc_lgb = print_evaluation(grid_lgb, X_test, y_test, "LightGBM")

# --------------------------
# 4. 使用 TPOT 进行自动化建模（可选）
# --------------------------
# 定义一个只包含以上四种算法的配置字典
custom_config = {
    'sklearn.linear_model.LogisticRegression': {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear'],
        'max_iter': [1000]
    },
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    },
    'xgboost.XGBClassifier': {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'use_label_encoder': [False],
        'eval_metric': ['mlogloss']
    },
    'lightgbm.LGBMClassifier': {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'num_leaves': [31, 50]
    }
}

print("\n运行 TPOT 进行自动化建模 ...")
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    cv=cv,
    random_state=42,
    verbosity=2,
    config_dict=custom_config,
    scoring='roc_auc_ovo',
    n_jobs=-1
)
tpot.fit(X_train, y_train)
tpot_score = tpot.score(X_test, y_test)
print(f"TPOT 最佳流水线在测试集上的得分: {tpot_score:.4f}")

# 如果需要，可将 TPOT 的最佳流水线导出为 Python 脚本
tpot.export('tpot_diabetes_pipeline.py')

# --------------------------
# 5. 模型选择与保存
# --------------------------
# 这里可根据上面的评估结果选择最佳模型。这里示例选择 TPOT 的流水线作为最终模型
best_model = tpot.fitted_pipeline_
joblib.dump(best_model, 'best_diabetes_model.pkl')
print("最佳模型已保存至 best_diabetes_model.pkl")
