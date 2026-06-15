import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure required NLTK resources are downloaded
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download nltk resource '{res}': {str(e)}")

download_nltk_resources()

def clean_text_fallback(text):
    """
    Fallback regex cleaning if NLTK tokenizer has issues.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    return tokens

def main():
    print("=== NLP Day 1: Text Processing ===")
    
    text = "The quick brown foxes are jumping over the lazy dogs. Running fast is exciting, but resting is also important!"
    print(f"Original Text:\n  '{text}'\n")
    
    # 1. Sentence and Word Tokenization
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        print(f"Sentence Tokenization (NLTK): {sentences}")
        print(f"Word Tokenization (NLTK): {words[:10]}...")
    except Exception as e:
        print("Sentence/Word tokenization failed (using fallback regex split).")
        words = clean_text_fallback(text)
        print(f"Fallback Tokenization: {words[:10]}...")
    
    # 2. Convert to lowercase & clean punctuation
    cleaned_words = [w.lower() for w in words if w.isalnum()]
    print(f"Cleaned Words (lowercase & alphanumeric): {cleaned_words[:10]}...")
    
    # 3. Stop Words Removal
    try:
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in cleaned_words if w not in stop_words]
        print(f"Stop Words Removed: {filtered_words[:10]}...")
    except Exception as e:
        # Simple manual fallback stop words list if nltk resource not available
        manual_stops = {'the', 'is', 'are', 'and', 'but', 'over', 'to', 'a', 'an', 'in', 'on', 'of', 'for'}
        filtered_words = [w for w in cleaned_words if w not in manual_stops]
        print(f"Fallback Stop Words Removed: {filtered_words[:10]}...")
        
    # 4. Stemming (Porter Stemmer)
    # Reduces words to their word stem/base form (often crude)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in filtered_words]
    print(f"Stemmed Words (Porter Stemmer): {stemmed_words}")
    
    # 5. Lemmatization (WordNet Lemmatizer)
    # Uses vocabulary and morphological analysis to return actual dictionary form (lemma)
    try:
        lemmatizer = WordNetLemmatizer()
        # By default lemmatizes nouns. We can pass pos='v' to lemmatize verbs.
        lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]
        lemmatized_verbs = [lemmatizer.lemmatize(w, pos='v') for w in filtered_words]
        print(f"Lemmatized Words (Noun default): {lemmatized_words}")
        print(f"Lemmatized Verbs (Verb target)  : {lemmatized_verbs}")
    except Exception as e:
        print("Lemmatization failed due to missing WordNet resource.")

if __name__ == "__main__":
    main()
