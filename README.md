# Bank Personal Loan Prediction 🏦

Predicting personal loan acceptance using bank customer data. The project
performs exploratory data analysis (EDA), feature engineering, and builds a
logistic regression classifier to predict whether a customer will take a
personal loan.

## 📊 About the Data

- **Dataset:** `Bank_Personal_Loan_Modelling(1).csv` (5,000 rows × 14 columns)
- **Target:** `Personal Loan` (0: no loan, 1: took a loan)

| Column              | Description |
|---------------------|-------------|
| `ID`                | Customer ID |
| `Age`               | Customer age |
| `Experience`        | Years of professional experience |
| `Income`            | Annual income |
| `ZIP Code`          | Customer ZIP code |
| `Family`            | Family size (1–4) |
| `CCAvg`             | Average credit card spending |
| `Education`         | Education level (1/2/3) |
| `Mortgage`          | Mortgage value |
| `Personal Loan`     | Target — took a personal loan |
| `Securities Account`| Has a securities account (0/1) |
| `CD Account`        | Has a CD account (0/1) |
| `Online`            | Uses online banking (0/1) |
| `CreditCard`        | Uses a credit card (0/1) |

## 🧰 Tech Stack

- Python 3
- pandas, numpy
- matplotlib, seaborn, plotly
- scikit-learn (LogisticRegression, Naive Bayes, KNN, GridSearchCV)
- imbalanced-learn (SMOTE, RandomUnderSampler)

## 🔍 What This Project Does

1. **Data cleaning** — fixed negative `Experience` values (abs), converted
   `CCAvg` from string to float.
2. **Exploratory analysis** — distribution and relationship analysis of age,
   income, family size, education, mortgage, and bank services vs. loan status.
3. **Geospatial analysis** — merged with a ZIP-code dataset and plotted an
   interactive map of customers (Plotly + Mapbox/OpenStreetMap).
4. **Modeling** — standardized features, handled the class-imbalance problem
   (`Personal Loan` is only ~9.6% positive), and trained a **Logistic
   Regression** classifier (with additional models and tools imported).

## 💡 Key Findings

- Only about **10%** of customers took a loan — a class-imbalance problem.
- Almost only customers with **income ≥ 60** took a loan.
- Most customers have `CCAvg < 3`, but most borrowers have `CCAvg > 3`.
- Higher education increases the likelihood of taking a loan.
- Customers with zero/low mortgage are more likely to take a loan.
- Most customers are located in the **Los Angeles** and **San Francisco** areas.
- Age, online-banking and credit-card usage showed weak or unclear effects.
