import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Title
# -------------------------------
st.set_page_config(page_title="Loan Predictor", layout="wide")
st.title("🏦 Smart Loan Prediction System")

st.write("Predict whether a customer will buy a personal loan")

# -------------------------------
# Sample Dataset (Extended)
# -------------------------------
data = pd.DataFrame({
    "Age": [25, 35, 45, 23, 40, 50, 28],
    "Salary": [30000, 60000, 80000, 20000, 75000, 90000, 40000],
    "Experience": [1, 10, 20, 0, 15, 25, 3],
    "Loan": [0, 1, 1, 0, 1, 1, 0]
})

X = data[["Age", "Salary", "Experience"]]
y = data["Loan"]

# -------------------------------
# Train Model
# -------------------------------
model = DecisionTreeClassifier()
model.fit(X, y)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 60, 30)
salary = st.sidebar.slider("Salary", 10000, 100000, 50000)
experience = st.sidebar.slider("Experience (years)", 0, 30, 5)

# -------------------------------
# Prediction
# -------------------------------
st.subheader("Prediction Result")

if st.button("Predict"):
    input_data = [[age, salary, experience]]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"✅ Will Buy Loan (Confidence: {max(probability)*100:.2f}%)")
    else:
        st.error(f"❌ Will NOT Buy Loan (Confidence: {max(probability)*100:.2f}%)")

# -------------------------------
# Model Accuracy
# -------------------------------
st.subheader("Model Performance")
accuracy = model.score(X, y)
st.write(f"Accuracy: {accuracy*100:.2f}%")

# -------------------------------
# Data Visualization
# -------------------------------
st.subheader("Data Visualization")

fig, ax = plt.subplots()
ax.scatter(data["Age"], data["Salary"])
ax.set_xlabel("Age")
ax.set_ylabel("Salary")
st.pyplot(fig)

# -------------------------------
# Decision Tree Visualization
# -------------------------------
if st.checkbox("Show Decision Tree"):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(
        model,
        feature_names=["Age", "Salary", "Experience"],
        class_names=["No", "Yes"],
        filled=True,
        ax=ax
    )
    st.pyplot(fig)

# -------------------------------
# Batch Prediction (CSV Upload)
# -------------------------------
st.subheader("Batch Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    predictions = model.predict(df)
    df["Prediction"] = predictions

    st.write(df)