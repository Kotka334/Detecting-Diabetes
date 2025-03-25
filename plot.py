import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Read the cleaned dataset
df_cleaned = pd.read_csv('Cleaned_Diabetes_Dataset.csv')

# Load the pre-trained model
with open('best_diabetes_model.pkl', 'rb') as f:
    model = joblib.load(f)

# List of columns to plot
columns_to_plot = ['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# Define the prediction and plot function
def predict_and_plot(Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI):
    # Prepare the input data for prediction
    input_data = np.array([[Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI]])
    
    # Make the prediction using the model
    prediction = model.predict(input_data)[0]
    
    # Map the prediction to corresponding text
    if prediction == 0:
        prediction_text = "Not Diabetic"
    elif prediction == 1:
        prediction_text = "Diabetic"
    else:
        prediction_text = "Prediabetic"
    
    # Create a new figure for the boxplots
    plt.figure(figsize=(12, 8))
    
    # Dictionary for the input data to overlay on the boxplots
    input_data_dict = {
        "Urea": Urea,
        "Cr": Cr,
        "HbA1c": HbA1c,
        "Chol": Chol,
        "TG": TG,
        "HDL": HDL,
        "LDL": LDL,
        "VLDL": VLDL,
        "BMI": BMI,
    }
    
    # Loop through the columns to create subplots
    for i, column in enumerate(columns_to_plot, 1):
        plt.subplot(3, 3, i)
        # Draw the boxplot with the distribution from the dataset
        sns.boxplot(y=df_cleaned[column], color=sns.color_palette("Set2")[i % 8])
        plt.title(f"Boxplot of {column}")
        
        # Overlay the input data as a red dot at the correct position
        plt.scatter(0, input_data_dict[column], color="red", s=50, zorder=5)
        
        # Optionally, adjust x-limits so that the box appears centered
        plt.xlim(-0.5, 0.5)
    
    plt.tight_layout()
    
    # Return both prediction text and the plot
    return prediction_text, plt.gcf()

# Create Gradio interface

interface = gr.Interface(
    fn=predict_and_plot,
    inputs=[
        gr.Number(label="Gender (0 = Female, 1 = Male)"),
        gr.Number(label="Age"),
        gr.Number(label="Urea"),
        gr.Number(label="Cr"),
        gr.Number(label="HbA1c"),
        gr.Number(label="Chol"),
        gr.Number(label="TG"),
        gr.Number(label="HDL"),
        gr.Number(label="LDL"),
        gr.Number(label="VLDL"),
        gr.Number(label="BMI")
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Plot()
    ],
    live=False,  # live update disabled; user must click "Submit"
    title="Diabetes Prediction Model",
    description="The model predicts whether you are diabetic, prediabetic, or not diabetic. The red dot on each boxplot shows the position of your input data.",
    # Adding the markdown widget for additional information
    allow_flagging="never",
                        examples=[
                     [0,51,2.1,46.0,5.9,4.1,2.7,1.0,2.0,1.2,36.0],
                     [1,28,3.5,61.0,8.5,4.5,1.9,1.1,2.6,0.8,37.0],
                     [1,26,4.5,62.0,4.9,3.7,1.4,1.1,2.1,0.6,23.0],
                     [0,63,6.6,96.0,8.9,5.8,1.7,1.7,3.4,0.8,32.0]
                 ]
)

# Adding the info widget (Markdown Accordion) before the inputs and outputs
with interface:

    with gr.Accordion("📖 Click to Learn More", open=True):
        gr.Markdown("""
        This model predicts the risk of diabetes based on clinical parameters such as 
        gender, age, cholesterol levels, and BMI.  
        - Input your details below.  
        - Click 'Submit' to get a prediction.  
        - The model outputs a risk classification.
        - We then train multiple models, including Logistic Regression, Random Forest, XGBoost, and LightGBM, using GridSearchCV to find the best hyperparameters.

        To automate the model selection process, we use TPOT, which helps us identify the best model pipeline. After evaluating the models, we select the best-performing one and save it for future predictions.

        The final model can predict whether a person has diabetes based on their health information, providing valuable insights for healthcare professionals.
        """)

# Launch Gradio interface
interface.launch()
