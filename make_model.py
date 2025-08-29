import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load Pima Indians Diabetes dataset
df = pd.read_csv(r"D:\ML_Model\disease_app\diabetes.csv")
# Features & target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as diabetes_model.pkl")
