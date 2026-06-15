import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def main():
    print("=== NLP Day 2: Vectorization ===")
    
    # Sample corpus of documents
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "This dog is lazy, but that dog is active and quick.",
        "Fast learning is exciting.",
        "The fox is quick and the dog is lazy."
    ]
    
    print("Corpus:")
    for idx, doc in enumerate(corpus):
        print(f"  Doc {idx + 1}: '{doc}'")
    print()
    
    # 1. Bag of Words (Count Vectorizer)
    # Binary/Frequency count representation
    bow_vectorizer = CountVectorizer(stop_words='english')
    bow_matrix = bow_vectorizer.fit_transform(corpus)
    
    print("--- 1. Bag of Words (BoW) Representation ---")
    print(f"Vocabulary: {bow_vectorizer.get_feature_names_out()}")
    print("Vocabulary Mapping (Word to Index):")
    print(bow_vectorizer.vocabulary_)
    print("\nDense Count Matrix:")
    print(bow_matrix.toarray())
    print()
    
    # 2. TF-IDF (Term Frequency - Inverse Document Frequency)
    # Weighs terms based on importance: local frequency vs global frequency
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    print("--- 2. TF-IDF Representation ---")
    print(f"Vocabulary: {tfidf_vectorizer.get_feature_names_out()}")
    print("\nTF-IDF Weight Matrix:")
    # Formatted to 4 decimal places for readability
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=tfidf_vectorizer.get_feature_names_out(),
        index=[f"Doc {i+1}" for i in range(len(corpus))]
    )
    print(tfidf_df.round(4))
    print()
    
    # 3. Vectorization with N-Grams
    # Capturing word sequence context (Bi-grams)
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    ngram_matrix = ngram_vectorizer.fit_transform(corpus)
    
    print("--- 3. N-Grams (Uni-grams & Bi-grams) ---")
    print("Features (vocabulary) including Bi-grams:")
    print(ngram_vectorizer.get_feature_names_out()[:15]) # Print first 15 features
    print(f"Total features extracted: {len(ngram_vectorizer.get_feature_names_out())}")

if __name__ == "__main__":
    main()
