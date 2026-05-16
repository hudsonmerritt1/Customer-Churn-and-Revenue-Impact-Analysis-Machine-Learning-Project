#Customer Churn Project

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

#load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.xls")

#data exploration
print("\n===== A Look at the Data =====")
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

#clean data
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = df.drop(columns=["customerID"])

#create variable for financial impact (6 months of charges)
df["EstimatedRevenueLoss"] = df["MonthlyCharges"] * 6

#graphs based on data pre-processing
plt.figure()
df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Count")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

plt.figure()
df.groupby("Contract")["Churn"].mean().plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate")
plt.show()

plt.figure()
df.groupby("Churn")["MonthlyCharges"].mean().plot(kind="bar")
plt.title("Average Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Average Monthly Charges")
plt.show()

#make dummay variables
df_model = pd.get_dummies(df, drop_first=True)

#define variables
X = df_model.drop(columns=["Churn", "EstimatedRevenueLoss"])
y = df_model["Churn"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

#scale data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

#function to assist with model evaluation
def evaluate_model(model_name, y_true, y_pred):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

#basic logistic regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_valid_scaled)

print("\n===== Logistic Regression Confusion Matrix =====")
print(confusion_matrix(y_valid, log_pred))

print("\n===== Logistic Regression Classification Report =====")
print(classification_report(y_valid, log_pred))

#decision tree model
tree_params = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_leaf": [10, 20, 50]
}

tree_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    tree_params,
    cv=5,
    scoring="recall"
)

tree_grid.fit(X_train, y_train)

tree_model = tree_grid.best_estimator_
tree_pred = tree_model.predict(X_valid)

print("\n===== Decision Tree Best Parameters =====")
print(tree_grid.best_params_)

print("\n===== Decision Tree Confusion Matrix =====")
print(confusion_matrix(y_valid, tree_pred))

print("\n===== Decision Tree Classification Report =====")
print(classification_report(y_valid, tree_pred))

#neural network model
nn_model = MLPClassifier(
    hidden_layer_sizes=(5, 4),
    max_iter=1000,
    random_state=42
)

nn_model.fit(X_train_scaled, y_train)

nn_pred = nn_model.predict(X_valid_scaled)

print("\n===== Neural Network Confusion Matrix =====")
print(confusion_matrix(y_valid, nn_pred))

print("\n===== Neural Network Classification Report =====")
print(classification_report(y_valid, nn_pred))

#random forest model
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [5, 10, 20]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring="recall"
)

rf_grid.fit(X_train, y_train)

rf_model = rf_grid.best_estimator_
rf_pred = rf_model.predict(X_valid)

print("\n===== Random Forest Best Parameters =====")
print(rf_grid.best_params_)

print("\n===== Random Forest Confusion Matrix =====")
print(confusion_matrix(y_valid, rf_pred))

print("\n===== Random Forest Classification Report =====")
print(classification_report(y_valid, rf_pred))

#compare model results
results = []

results.append(evaluate_model("Logistic Regression", y_valid, log_pred))
results.append(evaluate_model("Decision Tree", y_valid, tree_pred))
results.append(evaluate_model("Neural Network", y_valid, nn_pred))
results.append(evaluate_model("Random Forest Ensemble", y_valid, rf_pred))

results_df = pd.DataFrame(results)

print("\n===== Model Comparison (Accuracy, Precision, Recall, F1) =====")
print(results_df)

#variable importance in random forest model
importance_df = pd.DataFrame({
    "Variable": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n===== Top 15 Most Important Features (Random Forest) =====")
print(importance_df.head(15))

#variable selection with lasso
lasso = LassoCV(
    cv=5,
    random_state=42,
    max_iter=10000,
    tol=0.01
)

lasso.fit(X_train_scaled, y_train)

lasso_selected = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": lasso.coef_
})

lasso_selected = lasso_selected[lasso_selected["Coefficient"] != 0]

lasso_selected.sort_values(by="Coefficient", ascending=False)

print("\n===== Lasso Selected Variables =====")
print(lasso_selected.sort_values(by="Coefficient", ascending=False))

#revenue impact analysis with the best model
best_model = rf_model

churn_probabilities = best_model.predict_proba(X_valid)[:, 1]

revenue_analysis = X_valid.copy()
revenue_analysis["ActualChurn"] = y_valid.values
revenue_analysis["ChurnProbability"] = churn_probabilities
revenue_analysis["MonthlyCharges"] = df.loc[X_valid.index, "MonthlyCharges"]
revenue_analysis["EstimatedRevenueLoss"] = df.loc[X_valid.index, "EstimatedRevenueLoss"]

revenue_analysis["ExpectedRevenueAtRisk"] = (
    revenue_analysis["ChurnProbability"] * revenue_analysis["EstimatedRevenueLoss"]
)

revenue_analysis = revenue_analysis.sort_values(
    by="ExpectedRevenueAtRisk",
    ascending=False
)

print("\n===== Top 20 Customers by Expected Revenue at Risk =====")
print(revenue_analysis.head(20))

#revenue at risk
top_10_percent = revenue_analysis.head(int(len(revenue_analysis) * 0.10))

total_revenue_at_risk = revenue_analysis["ExpectedRevenueAtRisk"].sum()
top_10_revenue_at_risk = top_10_percent["ExpectedRevenueAtRisk"].sum()

print("\n===== Revenue Risk Summary =====")
print("Total Expected Revenue at Risk:", round(total_revenue_at_risk, 2))
print("Top 10% Expected Revenue at Risk:", round(top_10_revenue_at_risk, 2))
print("Percent of Revenue Risk from Top 10%:",
      round((top_10_revenue_at_risk / total_revenue_at_risk) * 100, 2), "%")

#revenue at risk graph
plt.figure()
revenue_analysis["ExpectedRevenueAtRisk"].head(25).plot(kind="bar")
plt.title("Top 25 Customers by Expected Revenue at Risk")
plt.xlabel("Customer Index")
plt.ylabel("Expected Revenue at Risk")
plt.show()






















































