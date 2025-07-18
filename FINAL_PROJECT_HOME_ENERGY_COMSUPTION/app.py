import gradio as gr
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('energy_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names from the CSV (assuming the first row contains headers)
df = pd.read_csv('home_energy.csv', nrows=1)
feature_names = list(df.columns)

# Remove the target column if present (assuming last column is target)
if 'target' in feature_names:
    feature_names.remove('target')

# Define the prediction function
def predict_energy(*inputs):
    X = np.array(inputs).reshape(1, -1)
    prediction = model.predict(X)[0]
    return prediction

# Create Gradio interface
demo = gr.Interface(
    fn=predict_energy,
    inputs=[gr.Number(label=feat) for feat in feature_names],
    outputs=gr.Number(label="Predicted Energy Consumption"),
    title="Home Energy Consumption Predictor",
    description="Enter the values for each feature to predict energy consumption."
)

demo.launch()
