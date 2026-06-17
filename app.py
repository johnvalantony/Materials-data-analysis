import gradio as gr
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = {
    "Material": [
        "Cobalt Tungstate", "Silicon", "Titanium Dioxide",
        "Zinc Oxide", "Gallium Nitride", "Copper Oxide",
        "Iron Oxide", "Cadmium Sulfide", "Zinc Sulfide", "Tin Oxide"
    ],
    "Bandgap_eV": [2.8, 1.1, 3.2, 3.4, 3.4, 1.2, 2.2, 2.4, 3.6, 3.6],
    "Application": [
        "Photocatalysis", "Solar cells", "Photocatalysis",
        "LEDs", "LEDs", "Solar cells",
        "Batteries", "Solar cells", "Phosphors", "Gas sensors"
    ]
}

df = pd.DataFrame(data)
X = df[["Bandgap_eV"]]
y = df["Application"]

model = DecisionTreeClassifier()
model.fit(X, y)

def predict_application(bandgap):
    prediction = model.predict([[bandgap]])[0]
    return f"🔬 Predicted Application: {prediction}"

demo = gr.Interface(
    fn=predict_application,
    inputs=gr.Slider(0.5, 4.5, value=2.0, label="Bandgap (eV)"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Materials Application Predictor",
    description="Built by John Valan Tony | M.Sc Physics | Predicts material application from bandgap value using Machine Learning"
)

demo.launch()
