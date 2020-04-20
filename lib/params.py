class Params():
    hidden_dim = 128
    embedding_dim = 50
    bidirectional = False
    num_layers = 1
    dropout = 0.3
    embeddings_processed_weights = 'model/embeddings_weights.json'
    embeddings_path = 'model/glove.6B.50d.txt'
    #embeddings_path = None
    device = "cuda"
    window_size = 100
    window_shift = 100
    min_freq = 2
    max_freq = 0
    vocabulary_path = 'model/vocabulary.json'
    label_vocabulary_path = 'model/label_vocabulary.json'
    stopwords_path = "model/stopwords.json"
