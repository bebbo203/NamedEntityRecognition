class Params():
    hidden_dim = 256
    embedding_dim = 50
    bidirectional = True
    num_layers = 2
    dropout = 0.3
    embeddings_processed_weights = 'model/embeddings_weights.json'
    embeddings_path = 'model/glove.6B.50d.txt'
    device = "cuda"
    window_size = 50
    window_shift = 50
    min_freq = 0
    max_freq = 0
    vocabulary_path = 'model/vocabulary.json'
    label_vocabulary_path = 'model/label_vocabulary.json'
    stopwords_path = None
    #Comment