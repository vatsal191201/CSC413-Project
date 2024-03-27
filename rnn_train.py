import tensorflow as tf
import numpy as np
import os
import pickle
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Directory for storing pickled files
pickle_dir = 'pkl'
os.makedirs(pickle_dir, exist_ok=True)

### Data Preprocessing ###

def preprocess_lyrics(file_path):
    """Read and preprocess lyrics data."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower().replace('\n', ' \n ')
    text = ''.join([char for char in text if char not in punctuation])
    words = text.split(' ')
    vocab = sorted(set(words))
    print(f'\nThere are {len(vocab)} unique words in the lyrics file.')
    return words, vocab

def save_pickle(data, file_name):
    """Save data to a pickle file."""
    with open(os.path.join(pickle_dir, file_name), 'wb') as file:
        pickle.dump(data, file)

# Modify path accordingly
words, vocab = preprocess_lyrics('drake_lyrics.txt')
save_pickle(vocab, 'vocab.pkl')

### Word Mapping ###

word2idx = {u: i for i, u in enumerate(vocab)}
idx2word = np.array(vocab)
words_as_int = np.array([word2idx[word] for word in words])

save_pickle(word2idx, 'word2idx.pkl')
save_pickle(idx2word, 'idx2word.pkl')

### Dataset Preparation ###

seq_length = 100
examples_per_epoch = len(words) // (seq_length + 1)

word_dataset = tf.data.Dataset.from_tensor_slices(words_as_int)
sequences = word_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    """Split data into input and target text."""
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)
dataset_sb = dataset.shuffle(10000).batch(64, drop_remainder=True)

### Model Building ###

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """Build and compile the RNN model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(units=rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model_params = [vocab_size, embedding_dim, rnn_units]
save_pickle(model_params, 'model_params.pkl')

rnn = build_model(vocab_size, embedding_dim, rnn_units, 64)

### Compile the Model ###

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

rnn.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

### Model Training ###

checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'ckpt_{epoch}'),
    save_weights_only=True
)

rnn.fit(dataset_sb, epochs=10, callbacks=[checkpoint_callback])
