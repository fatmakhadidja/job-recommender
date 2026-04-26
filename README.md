# CareerAI — ML Job Recommendation System

A Flask web application that predicts your ideal tech job role based on your skills and experience level, powered by a Random Forest classifier trained on a balanced synthetic dataset.

---

## Demo

Select your programming languages, frameworks, and experience level → get an instant prediction with confidence score and full probability breakdown across all 6 roles.
<img width="1336" height="616" alt="image" src="https://github.com/user-attachments/assets/054c12d6-f396-4fc1-a5d5-91cd82a3545e" />

<img width="1345" height="610" alt="image" src="https://github.com/user-attachments/assets/63812991-b325-4f03-be8b-cec61665e28e" />

<img width="1348" height="606" alt="image" src="https://github.com/user-attachments/assets/0ab2e669-0449-462e-82be-9cc8b23d741d" />

---

## Project Structure

```
job_recommender/
├── app.py                  # Flask backend — routes and prediction logic
├── model.pkl               # Pre-trained Random Forest model (joblib)
├── requirements.txt        # Python dependencies
├── templates/
│   ├── index.html          # Skill selection form
│   └── result.html         # Prediction result page
└── static/
    └── style.css           # Custom design system (dark theme, no Bootstrap)
```

---

## Quickstart

**1. Clone or unzip the project**
```bash
cd job_recommender
```

**2. (Optional) Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
python app.py
```

**5. Open your browser**
```
http://localhost:5000
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Flask | ≥ 3.0.0 | Web framework |
| scikit-learn | ≥ 1.4.0 | Logistic regression model |
| joblib | ≥ 1.3.0 | Model serialization |
| numpy | ≥ 1.26.0 | Numerical operations |
| pandas | ≥ 2.0.0 | Feature DataFrame construction |

---

## How It Works

### Input Features (17 total)

**Programming Languages** (binary 0/1): `python`, `java`, `dart`, `c`, `javascript`, `html`, `css`, `php`

**Frameworks** (binary 0/1): `django`, `flask`, `spring`, `react`, `angular`, `nodejs`, `flutter`, `laravel`

**Experience Level** (numeric): `0` = Junior · `1` = Mid-Level · `2` = Senior · `3` = Expert

### Prediction Pipeline

```
Form checkboxes → pandas DataFrame → model.predict() + model.predict_proba()
                                          ↓
                              Predicted role + confidence %
                              + ranked probability for all 6 roles
```

### Output Classes

| Label | Job Role |
|---|---|
| 0 | Backend Developer |
| 1 | Frontend Developer |
| 2 | Full Stack Developer |
| 3 | Mobile Developer |
| 4 | Data Scientist |
| 5 | DevOps Engineer |

---

## The Model

- **Algorithm:** logistic regression
- **Training data:** 360 rows, 60 per class — pattern-based synthetic dataset
- **Test accuracy:** 98.1%
- **Serialized with:** `joblib`

The training dataset was generated with realistic skill patterns per role (e.g. Dart + Flutter → Mobile Developer, HTML + CSS + React → Frontend Developer) rather than random binary assignment, which gives the model strong class separation.

### Retraining the Model

If you want to retrain on new data:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('your_dataset.csv')

FEATURES = [
    'python', 'java', 'dart', 'c', 'javascript', 'html', 'css', 'php',
    'django', 'flask', 'spring', 'react', 'angular', 'nodejs', 'flutter',
    'laravel', 'experience_level'
]

X = df[FEATURES]
y = df['job_role']

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

joblib.dump(model, 'model.pkl')
```

---

## Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Renders the skill selection form |
| `POST` | `/predict` | Accepts form data, returns result page |

---

## UI Features

- Dark tech theme with amber accent color system
- Skill chips with animated selection state
- SVG confidence ring on result page
- Probability bar chart for all 6 roles
- Skill summary card (selected languages, frameworks, experience)
- Fully responsive layout (mobile-friendly)

---

## Extending the App

**Add a REST API endpoint** — return JSON instead of HTML for use with a frontend framework:

```python
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    X = pd.DataFrame([data])[FEATURES]
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    return {'role': JOB_ROLES[pred], 'probabilities': proba}
```

**Swap the model** — replace `model.pkl` with any scikit-learn compatible classifier (XGBoost, SVM, etc.) that exposes `predict()` and `predict_proba()`. No other code changes needed.
