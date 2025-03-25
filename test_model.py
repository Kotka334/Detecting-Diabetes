import joblib
import pandas as pd

# 加载模型（确保 best_diabetes_model.pkl 文件与此脚本在同一目录下）
model = joblib.load('best_diabetes_model.pkl')

# 构造一个测试样例，确保特征名称和训练时一致
test_data = pd.DataFrame({
    'Gender': [1],   # 如果训练时用的是 "Male"/"Female"，请保持一致
    'AGE': [34],
    'Urea': [3.9],
    'Cr': [81],
    'HbA1c': [6],
    'Chol': [6.2],
    'TG': [3.9],
    'HDL': [0.8],
    'LDL': [1.9],
    'VLDL': [1.8],
    'BMI': [23]
})

# 使用模型进行预测
prediction = model.predict(test_data)
print("预测结果：", prediction)
