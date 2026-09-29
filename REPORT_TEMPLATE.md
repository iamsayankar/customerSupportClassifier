# Customer Support Ticket Classification System
## AI Fresher – Technical Assignment

### A. Problem Understanding
The objective is to build a basic machine-learning system that predicts the category of a customer support ticket from its description.

### B. Dataset
A self-created CSV dataset was prepared with 200 sample records and 5 categories:
Login Issue, Application Error, Report, Account Update, and Performance.

Fields:
ticket_id, ticket_description, category, priority, status.

### C. Data Preprocessing
The dataset is loaded with Pandas. Missing values and duplicates are checked. Duplicate records are removed where present. Ticket descriptions are converted to lowercase, unnecessary characters are removed, and extra spaces are normalized.

### D. Model Selection
The selected approach is TF-IDF with Logistic Regression.

TF-IDF converts text into numerical features. Logistic Regression then learns patterns in those features and predicts the ticket category.

### E. Training and Testing
The dataset is divided into 80% training data and 20% testing data. The model is trained on the training portion and evaluated on unseen test data.

### F. Results
Run:
`python src/train.py`

Then copy the generated Accuracy, Precision, Recall, F1 Score and the confusion matrix into this section. Do not invent metric values.

### G. Prediction
Use:
`python src/predict.py`

Or run the Streamlit interface:
`streamlit run src/app.py`

Take a screenshot of at least 5 new-ticket predictions and place it in `screenshots/prediction.png`.

### H. Limitations
The dataset is relatively small and contains a limited number of categories. Real customer tickets may use ambiguous or unexpected wording.

### I. Future Improvements
A larger dataset, more categories, stronger preprocessing, model comparison, database integration, continuous retraining, human feedback, production APIs, and advanced AI methods could improve the system.
