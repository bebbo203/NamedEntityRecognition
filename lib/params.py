class Params():
    hidden_dim = 512
    embedding_dim = 50
    char_embedding_dim = 10
    single_char_embedding_dim = 4
    alphabet_size = 107
    bidirectional = True
    num_layers = 2
    dropout = 0.5
    embeddings_processed_weights = 'model/embeddings_weights.json'
    embeddings_path = 'model/glove.6B.50d.txt'
    max_word_lenght = 16
    device = "cuda"
    window_size = 5
    window_shift =  1
    min_freq = 5
    max_freq = 0
    vocabulary_path = 'model/vocabulary.json'
    label_vocabulary_path = 'model/label_vocabulary.json'
    stopwords_path = "model/stopwords.json"
    #stopwords_path = None