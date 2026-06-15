import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def main():
    print("=== NLP Day 4: Sentiment Analysis ===")
    
    # 1. Generate Synthetic Product Review Dataset (Positive & Negative reviews)
    positive_reviews = [
        "excellent product, works perfectly and highly recommend!",
        "absolutely love this device, best purchase I have made this year.",
        "great customer service, prompt response and very helpful.",
        "simple setup, very user friendly interface and sleek design.",
        "very high quality materials, durable and sturdy structure.",
        "delivery was fast and package arrived in perfect condition.",
        "worth every penny, makes tasks much easier and saves time.",
        "the battery life is amazing, lasts for several days without charging.",
        "functions exactly as described, exceeded my expectations completely.",
        "beautiful display screen, crystal clear images and bright colors."
    ] * 5  # 50 samples
    
    negative_reviews = [
        "terrible product, stopped working after two days of use.",
        "do not buy this item, complete waste of money and time.",
        "poor customer support, unhelpful agents and long waiting times.",
        "extremely difficult setup process, user manual is confusing.",
        "very cheap materials used, broke easily on first usage.",
        "shipment was delayed, packaging arrived damaged and open.",
        "overpriced and useless, does not match description features.",
        "battery drain is fast, needs to be plugged in constantly.",
        "defective screen pixels, blurry visual output and screen glitches.",
        "disappointing experience, lacks key features and runs slowly."
    ] * 5  # 50 samples
    
    reviews = positive_reviews + negative_reviews
    # Labels: 1 = Positive, 0 = Negative
    labels = [1]*50 + [0]*50
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # 2. Build Pipeline (TF-IDF Vectorizer + Logistic Regression Classifier)
    sentiment_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # 3. Train Pipeline
    sentiment_pipeline.fit(X_train, y_train)
    
    # 4. Model Evaluation
    y_pred = sentiment_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Sentiment Classifier Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    
    # 5. Predict on Unseen Customer Feedback
    new_feedbacks = [
        "Fast delivery and a wonderful experience overall!",
        "It is slow, expensive, and broke down almost immediately.",
        "The battery lasts longer than my old one, but the design is heavy.",
        "Terrible instructions. I returned it back to the seller."
    ]
    
    predictions = sentiment_pipeline.predict(new_feedbacks)
    probabilities = sentiment_pipeline.predict_proba(new_feedbacks)
    
    print("\n--- Predictions on New Feedback ---")
    for i, feedback in enumerate(new_feedbacks):
        pred_label = "Positive" if predictions[i] == 1 else "Negative"
        conf = probabilities[i][predictions[i]] * 100
        print(f"Feedback: '{feedback}'")
        print(f"  Result: {pred_label} (Confidence: {conf:.2f}%)\n")
        
    # 6. Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Sentiment Classification Confusion Matrix")
    
    import os
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "nlp_day_04_sentiment_cm.png")
    # Wait, the current execution directory is ML_Model_Deployment, so plots will go to c:\Users\KondasaniBalaji\ML Projects\ML_Model_Deployment\plots\nlp_day_04_sentiment_cm.png
    # Let's save it.
    plt.savefig(plot_path)
    print(f"Saved confusion matrix visualization to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
