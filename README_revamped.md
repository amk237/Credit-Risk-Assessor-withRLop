# Credit Risk Assessment Agent
### Explainable credit risk prediction with a baseline ML model, structured reasoning, and a reinforcement learning layer

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-orange">
  <img alt="Status" src="https://img.shields.io/badge/Status-Research%20Project-success">
  <img alt="Notebook" src="https://img.shields.io/badge/Format-Jupyter%20Notebook-informational">
</p>

## Overview

This project explores how machine learning and explainability techniques can be combined for **credit risk assessment**. It predicts whether a loan application is likely to be risky using a trained classification pipeline, then extends that baseline with:

- a **structured reasoning layer** that explains how the model evaluates an applicant
- an **RL-inspired decision layer** that experiments with adaptive actions on top of the base prediction

The goal is not just to make a prediction, but to make the decision process easier to inspect, discuss, and improve.

---

## Why this project?

Credit decisions are high-impact. A model that outputs only **approve** or **deny** is useful, but a model that also gives a **clear rationale** is much more valuable for learning, debugging, and stakeholder trust.

This project was built to answer a simple question:

> Can we make credit risk prediction more understandable without giving up practical ML performance?

---

## What the project does

The notebook builds and compares three agents:

### 1) Baseline Credit Risk Agent
A traditional machine learning pipeline using:

- **feature preprocessing**
- **one-hot encoding** for categorical variables
- **standardization** for numeric variables
- a **Random Forest classifier**

This serves as the core predictive model.

### 2) CoT-Style Credit Risk Agent
This version adds a **step-by-step reasoning process** on top of the baseline model.  
It walks through factors such as:

- applicant age
- income
- home ownership
- employment length
- loan amount
- interest rate
- prior defaults
- credit history length

This makes the final output feel more interpretable and human-readable.

### 3) RL-CoT Credit Risk Agent
This version experiments with a simple **reinforcement-learning-style layer** using a Q-table to adapt actions from state-like application features.

It is an exploratory extension meant to test whether adaptive decision logic can complement the baseline model and reasoning layer.

---

## Model pipeline

```text
Raw applicant data
        ↓
Feature selection
        ↓
Preprocessing
  • StandardScaler for numeric columns
  • OneHotEncoder for categorical columns
        ↓
RandomForestClassifier
        ↓
Risk probability
        ↓
Decision + explanation
        ↓
(Optional) RL-style action refinement
```

---

## Features used

The current model uses the following applicant and loan features:

- `person_age`
- `person_income`
- `person_home_ownership`
- `person_emp_length`
- `loan_intent`
- `loan_amnt`
- `loan_int_rate`
- `cb_person_default_on_file`
- `cb_person_cred_hist_length`

**Target variable:**
- `loan_status`

---

## Results

The project reports strong baseline performance on the dataset.

### Example evaluation
- **Accuracy:** ~0.91

### Example class-level performance
- strong precision and recall for lower-risk / denial-detection patterns
- comparatively weaker recall on the minority class, suggesting room for improvement in class balance handling

This is a good starting point for an educational or experimental credit modeling workflow.

---

## Example reasoning output

```text
CoT Agent Decision: Low Risk - Approve Loan

Thought Process:
- Applicant age: 30 years. Employment length: 5 years.
- Annual income: $60,000. Home ownership: RENT.
- Loan amount: $10,000. Purpose: PERSONAL. Interest rate: 10.0%.
- Default on file: N. Credit history length: 5 years.
- Calculated risk probability: 0.05. Low risk.
```

This kind of output is useful because it shows *why* the model reached its conclusion, not just *what* the conclusion was.

---

## Repository structure

```bash
.
├── credit_worthiness.ipynb      # Main notebook with training, evaluation, and agent logic
├── credit_risk_dataset.csv      # Dataset used for experimentation
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## Tech stack

- **Python**
- **pandas**
- **numpy**
- **scikit-learn**
- **matplotlib**
- **Jupyter Notebook**

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
jupyter notebook
```

Open `credit_worthiness.ipynb` and run the cells in order.

---

## Key takeaways

- A standard ML pipeline can perform well on tabular credit data.
- Adding a reasoning layer improves interpretability and presentation.
- RL-style decision logic is an interesting experimental extension, though it still needs deeper validation.
- This project is best viewed as an **educational / research prototype**, not a production lending system.

---

## Limitations

This project has a few important limitations:

- it is trained on a single public dataset
- class imbalance may affect minority-class performance
- the RL component is exploratory rather than production-grade
- explanations are structured and human-readable, but they are not the same as formal regulatory model interpretability
- no fairness, bias, or compliance audit is included yet

For a real-world lending workflow, those issues would need serious attention.

---

## Future improvements

- Handle class imbalance more explicitly
- Add hyperparameter tuning
- Compare against additional models such as XGBoost or logistic regression
- Add fairness evaluation and bias checks
- Build a lightweight web interface for live applicant testing
- Improve the RL formulation with a more rigorous state/action/reward design
- Add SHAP or other post-hoc explainability tools for richer interpretation

---

## Dataset

This project uses the **Credit Risk Dataset** from Kaggle.

- Source: *Credit Risk Dataset* by laotse on Kaggle

Be sure to review the dataset license and usage terms before redistributing or using it in downstream work.

---

## Disclaimer

This repository is for **educational and research purposes only**.

It should **not** be used as the sole basis for real lending or underwriting decisions.  
Real-world credit systems require:

- fairness and bias evaluation
- regulatory compliance
- robust validation
- human oversight
- domain and legal review

---

## Author

**Muhammad Turner Gane**

- GitHub: [m-turnergane](https://github.com/m-turnergane)
- LinkedIn: [Muhammad Gane](https://www.linkedin.com/in/muhammad-gane/)
- Email: m.turnergane@gmail.com

---

## Final note

This project sits at the intersection of **machine learning**, **explainability**, and **decision systems**.  
If you're interested in interpretable AI, applied ML, or fintech experimentation, this repo is a solid place to start.

If you found it useful, consider starring the project.
