import joblib
import gradio as gr

# Load bundle
bundle = joblib.load("early_warning_binary_model.joblib")
# bundle = joblib.load(r"C:\Users\Sohail_Comp\Desktop\EarlyWarning\early_warning_binary_model.joblib")

model = bundle["model"]
threshold = bundle["decision_threshold"]


def predict(subject, message):

    text = f"{subject}\n{message}"

    probability = model.predict_proba([text])[0][1]

    prediction = int(probability >= threshold)

    if prediction == 1:
        label = " Early Warning"
    else:
        label = " Not Early Warning"

    return (
        label,
        f"{probability:.4f}"
    )


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Subject",
            placeholder="Enter email subject"
        ),
        gr.Textbox(
            label="Message",
            lines=10,
            placeholder="Enter email body"
        )
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability")
    ],
    title="Early Warning Email Classifier",
    description="Binary Classification using TF-IDF + Logistic Regression"
)

if __name__ == "__main__":
    demo.launch()