import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Vectorize Text
def vectorize_data(X_train, X_test, max_features=10000, ngram_range=(1, 2)):
    """
    Vectorizes text data using TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Train Model
def train_model(X_train, y_train, model_type, params=None):
    """
    Trains one of the four specified models.
    """
    if model_type == 'logistic':
        model = LogisticRegression(**(params or {}))
    elif model_type == 'naive_bayes':
        model = MultinomialNB(**(params or {}))
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**(params or {}))
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(**(params or {}))
    else:
        raise ValueError(f"Model type {model_type} is not supported.")
    
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints classification metrics and confusion matrix.
    """
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return cm

# Plot ROC Curve
def plot_roc_curve(model, X_test, y_test):
    """
    Plots the ROC curve for the model.
    """
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
