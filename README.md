
# Early Warning Email Classifier

A machine learning web application that classifies construction-related emails as **Early Warning** or **Not Early Warning** using **TF-IDF** feature extraction and **Logistic Regression**.

## Features

- Binary email classification
- TF-IDF (Word + Character N-grams)
- Logistic Regression classifier
- Probability score prediction
- Interactive Gradio web interface
- Trained using Scikit-learn

## Tech Stack

- Python
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Gradio
- Joblib

## Project Structure

```
EarlyWarning/
│
├── app.py
├── requirements.txt
├── early_warning_binary_model.joblib
├── README.md
```

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

The application will start locally at:

```
http://127.0.0.1:7860
```

## Model

- Feature Extraction: TF-IDF
- Classifier: Logistic Regression
- Output:
  - Early Warning
  - Not Early Warning

## Screenshots

(Add screenshots of your Gradio interface here.)

## Author

**Rimsha Kanwal**

GitHub: https://github.com/RimshaKanwal0
