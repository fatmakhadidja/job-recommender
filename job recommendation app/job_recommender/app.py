from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

FEATURES = [
    'python', 'java', 'dart', 'c', 'javascript', 'html', 'css', 'php',
    'django', 'flask', 'spring', 'react', 'angular', 'nodejs', 'flutter', 'laravel',
    'experience_level'
]

JOB_ROLES = {
    0: 'Backend Developer',
    1: 'Frontend Developer',
    2: 'Full Stack Developer',
    3: 'Mobile Developer',
    4: 'Data Scientist',
    5: 'DevOps Engineer',
}

JOB_ICONS = {
    0: '⚙️', 1: '🎨', 2: '🔗', 3: '📱', 4: '📊', 5: '🚀'
}

JOB_DESC = {
    0: 'You excel at building robust server-side systems, APIs, and databases.',
    1: 'You craft beautiful, interactive user interfaces and web experiences.',
    2: 'You bridge the gap between client and server with end-to-end expertise.',
    3: 'You build native and cross-platform mobile applications.',
    4: 'You extract insights from data using statistical and ML techniques.',
    5: 'You automate infrastructure, CI/CD pipelines, and cloud deployments.',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    row = {f: int(form.get(f, 0)) for f in FEATURES[:-1]}
    row["experience_level"] = int(form.get("experience_level", 0))

    X = pd.DataFrame([row])[FEATURES]
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    confidence = round(float(proba[pred]) * 100, 1)

    all_roles = sorted(
        [(JOB_ROLES[i], round(float(p) * 100, 1)) for i, p in enumerate(proba)],
        key=lambda x: -x[1]
    )

    selected_langs = [f for f in FEATURES[:8] if int(form.get(f, 0))]
    selected_frameworks = [f for f in FEATURES[8:16] if int(form.get(f, 0))]
    exp_map = {0: 'Junior', 1: 'Mid-Level', 2: 'Senior', 3: 'Expert'}
    exp_label = exp_map[int(form.get('experience_level', 0))]

    return render_template(
        'result.html',
        job_role=JOB_ROLES[pred],
        job_icon=JOB_ICONS[pred],
        job_desc=JOB_DESC[pred],
        confidence=confidence,
        all_roles=all_roles,
        selected_langs=selected_langs,
        selected_frameworks=selected_frameworks,
        exp_label=exp_label,
    )

if __name__ == '__main__':
    app.run(debug=True)
