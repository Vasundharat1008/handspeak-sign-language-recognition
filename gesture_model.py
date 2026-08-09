import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib

# Load dataset
df = pd.read_csv("gesture_data.csv")

print(df.head())

# Features
X = df[["flex1","flex2","flex3","flex4","flex5"]]

# Labels
y = df["label"]

# Split dataset
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train,y_train)

# Predict
y_pred=model.predict(X_test)

# Accuracy
accuracy=accuracy_score(y_test,y_pred)

print("\nAccuracy :",accuracy)

print("\nConfusion Matrix")

print(confusion_matrix(y_test,y_pred))

print("\nClassification Report")

print(classification_report(y_test,y_pred))

# Feature Importance

print("\nFeature Importance")

for feature,importance in zip(X.columns,model.feature_importances_):
    print(feature,":",round(importance,4))

# Save Model

joblib.dump(model,"gesture_model.pkl")

print("\nModel Saved Successfully")
