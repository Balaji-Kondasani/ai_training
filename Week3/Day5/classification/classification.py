import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=== Day 3: NLP Classification ===")
    
    # 1. Generate Synthetic Text Classification Dataset
    # 3 categories: tech, sports, finance
    tech_texts = [
        "new software update released for database server",
        "best programming languages to learn web development",
        "quantum computing processor reaches new speeds",
        "artificial intelligence algorithms predict trends",
        "latest smartphone features high resolution camera",
        "developer builds app using flutter and python",
        "operating system patch fixes critical security bugs",
        "cloud computing architecture for scalable web systems",
        "hacker accesses secure server database details",
        "install linux packages using terminal script"
    ] * 5  # 50 samples
    
    sports_texts = [
        "football match scheduled for tomorrow evening stadium",
        "athlete wins gold medal in championship final event",
        "basketball league finals scores and season recap",
        "tennis player qualifies tournament semi final matches",
        "coach designs strategy for upcoming championship match",
        "runners prepare marathon event next weekend city",
        "cricket match delayed rain stadium pitches wet",
        "sports training routines for professional swimmers",
        "swimming pool dimensions for Olympic competition finals",
        "soccer player transfers to rival league club"
    ] * 5  # 50 samples
    
    finance_texts = [
        "stock market prices index drops following inflation news",
        "central bank interest rates increase loan payments",
        "invest in mutual funds stocks treasury bonds",
        "company quarterly financial revenue profit report statements",
        "cryptocurrency prices fluctuate Bitcoin trade updates",
        "saving account annual yield percent calculator online",
        "business taxes corporate tax deductions forms file",
        "investment portfolio asset allocation advice guide",
        "startup raises venture capital series funding round",
        "global trade tariff agreements impact business profits"
    ] * 5  # 50 samples
    
    texts = tech_texts + sports_texts + finance_texts
    # labels: 0=tech, 1=sports, 2=finance
    labels = [0]*50 + [1]*50 + [2]*50
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    # 2. Vectorization: TF-IDF
    # Convert training and test strings to numerical vectors
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Dataset Size: {len(texts)} samples")
    print(f"Vocabulary Size: {len(vectorizer.get_feature_names_out())} unique words")
    
    # 3. Train Naive Bayes Classifier
    # MultinomialNB is highly effective for text classification count features
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)
    
    # 4. Model Inference & Evaluation
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    target_names = ["tech", "sports", "finance"]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 5. Predict on New Sentences
    new_sentences = [
        "I want to write code in python and deploy to cloud server",
        "He scored a goal in the soccer final match",
        "Corporate profits increase after federal bank cuts interest rates"
    ]
    
    new_tfidf = vectorizer.transform(new_sentences)
    predictions = clf.predict(new_tfidf)
    
    print("\n--- Running Predictions on New Text ---")
    for text, pred in zip(new_sentences, predictions):
        print(f"Text: '{text}' -> Predicted: {target_names[pred]}")

if __name__ == "__main__":
    main()
