import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Paths
DATA_PATH = "data/tickets.csv"
MODEL_PATH = "model/ticket_classifier.pkl"

os.makedirs("model", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)


# --------------------------------
# 1. Load Dataset
# --------------------------------

df = pd.read_csv(DATA_PATH)

print("Total Tickets:", len(df))

print("\nCategory Distribution:")
print(df["category"].value_counts())

print("\nPriority Distribution:")
print(df["priority"].value_counts())

print("\nStatus Distribution:")
print(df["status"].value_counts())


# --------------------------------
# 2. Basic Cleaning
# --------------------------------

df = df.drop_duplicates()

df["ticket_description"] = (
    df["ticket_description"]
    .fillna("")
    .str.lower()
)


# --------------------------------
# 3. Category Chart
# --------------------------------

category_counts = df["category"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    category_counts.index,
    category_counts.values
)

plt.title("Ticket Category Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Tickets")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "screenshots/category_distribution.png"
)

plt.show()


# --------------------------------
# 4. Priority Chart
# --------------------------------

priority_counts = df["priority"].value_counts()

plt.figure(figsize=(7, 5))

plt.bar(
    priority_counts.index,
    priority_counts.values
)

plt.title("Ticket Priority Distribution")
plt.xlabel("Priority")
plt.ylabel("Number of Tickets")

plt.tight_layout()

plt.savefig(
    "screenshots/priority_distribution.png"
)

plt.show()


# --------------------------------
# 5. Input and Target
# --------------------------------

X = df["ticket_description"]

y = df["category"]


# --------------------------------
# 6. Train/Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# --------------------------------
# 7. ML Pipeline
# --------------------------------

model = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000
        )
    )
])


# --------------------------------
# 8. Train Model
# --------------------------------

model.fit(
    X_train,
    y_train
)


# --------------------------------
# 9. Prediction
# --------------------------------

y_pred = model.predict(X_test)


# --------------------------------
# 10. Evaluation
# --------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# --------------------------------
# 11. Confusion Matrix
# --------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Category")

plt.ylabel("Actual Category")

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "screenshots/confusion_matrix.png"
)

plt.show()


# --------------------------------
# 12. Save Model
# --------------------------------

joblib.dump(
    model,
    MODEL_PATH
)


print("\nModel saved successfully!")

print(
    "Location:",
    MODEL_PATH
)