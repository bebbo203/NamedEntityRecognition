class Params():
    hidden_dim = 64
    embedding_dim = 50
    bidirectional = True
    num_layers = 4
    dropout = 0.5
    embeddings_processed_weights = 'model/embeddings_weights.json'
    embeddings_path = 'model/glove.6B.50d.txt'
    
    device = "cuda"
    window_size = 5
    window_shift = 1
    min_freq = 5
    max_freq = 0
    vocabulary_path = 'model/vocabulary.json'
    label_vocabulary_path = 'model/label_vocabulary.json'
    stopwords_path = "model/stopwords.json"
