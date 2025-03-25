from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# 在应用启动时加载已训练好的模型
with open('best_diabetes_model.pkl', 'rb') as f:
    model = joblib.load(f)
print("Loaded model type:", type(model))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            # 从表单中获取数据，并转换为浮点数
            Gender = float(request.form['Gender'])
            AGE = float(request.form['AGE'])
            Urea = float(request.form['Urea'])
            Cr = float(request.form['Cr'])
            HbA1c = float(request.form['HbA1c'])
            Chol = float(request.form['Chol'])
            TG = float(request.form['TG'])
            HDL = float(request.form['HDL'])
            LDL = float(request.form['LDL'])
            VLDL = float(request.form['VLDL'])
            BMI = float(request.form['BMI'])

            # 构造一个 1x11 的输入数组
            input_features = np.array([[Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI]])

            # 使用模型进行预测
            prediction = model.predict(input_features)[0]
        except Exception as e:
            error = f"发生错误：{str(e)}"

    return render_template('index.html', prediction=prediction, error=error)


if __name__ == '__main__':
    app.run(debug=True)
