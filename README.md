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


**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python app.py
```

**4. Open your browser**
```
http://localhost:5000
```

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

---

## UI Features

- Dark tech theme with amber accent color system
- Skill chips with animated selection state
- SVG confidence ring on result page
- Probability bar chart for all 6 roles
- Skill summary card (selected languages, frameworks, experience)
- Fully responsive layout (mobile-friendly)

---

**Swap the model** — replace `model.pkl` with any scikit-learn compatible classifier (XGBoost, SVM, etc.) that exposes `predict()` and `predict_proba()`. No other code changes needed.
