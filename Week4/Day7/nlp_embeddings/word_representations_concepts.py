def generate_skipgram_pairs(sentence, window_size=2):
    """
    Generates Skip-gram training pairs: (target_word, context_word).
    """
    words = sentence.lower().split()
    pairs = []
    
    for idx, word in enumerate(words):
        # Determine context window indices bounds
        start = max(0, idx - window_size)
        end = min(len(words), idx + window_size + 1)
        
        for i in range(start, end):
            if i != idx:
                pairs.append((word, words[i]))
                
    return pairs

def main():
    print("=== Day 13: Word Representation Concepts ===")
    
    sentence = "deep learning models process natural language text"
    print(f"Sample Sentence: '{sentence}'")
    
    # Generate skipgram training pairs
    pairs = generate_skipgram_pairs(sentence, window_size=2)
    print(f"\nGenerated Skip-gram Pairs (window_size=2):")
    for pair in pairs[:10]:
        print(f"  Target: '{pair[0]:<8}' -> Context: '{pair[1]}'")
    print(f"Total pairs generated: {len(pairs)}\n")
    
    # Word Representation Models Comparison
    print("Core Word Representation Architectures:")
    print("  1. Word2Vec (Google, 2013):")
    print("     * CBOW (Continuous Bag of Words): Predicts target word given context words.")
    print("     * Skip-gram: Predicts context words given target word (better for rare words).")
    print("  2. GloVe (Stanford, 2014):")
    print("     * Global Vectors for Word Representation. Solves the issue that Word2Vec ignores global corpus statistics.")
    print("     * Combines local context windows with global matrix factorization of a word co-occurrence matrix.")
    print("  3. FastText (Facebook, 2016):")
    print("     * Represents words as bags of character n-grams (e.g. 'learning' as ['lea', 'ear', 'arn', ...]).")
    print("     * Crucial advantage: Handles Out-Of-Vocabulary (OOV) words by building vectors from character subparts.")

if __name__ == "__main__":
    main()
