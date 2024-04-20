import tensorflow as tf
import pickle
import os
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


### Load Model Parameters and Weights ###

def load_pickle(file_name):
    """Load data from a pickle file."""
    with open(file_name, 'rb') as file:
        return pickle.load(file)

pickle_dir = 'pkl'
vocab_size, embedding_dim, rnn_units = load_pickle(os.path.join(pickle_dir, 'model_params.pkl'))
word2idx = load_pickle(os.path.join(pickle_dir, 'word2idx.pkl'))
idx2word = load_pickle(os.path.join(pickle_dir, 'idx2word.pkl'))
vocab = load_pickle(os.path.join(pickle_dir, 'vocab.pkl'))

def build_model(vocab_size, embedding_dim, rnn_units):
    """Rebuild the model for text generation (batch_size=1)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[1, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units)
model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
model.build(tf.TensorShape([1, None]))

### Text Generation ###

def generate_text(model, start_string, num_generate=500, temperature=1.0):
    """Generate text given a start string."""
    input_eval = [word2idx[s] for s in start_string.split()]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.reset_states()
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2word[predicted_id])

    return ' '.join(text_generated)

### Input Handling and Text Generation ###

def clean_and_split_input(input_text):
    """Preprocess and validate the user's input text."""
    input_text = input_text.lower().translate(str.maketrans('', '', punctuation))
    words = input_text.split()
    non_vocab_words = [word for word in words if word not in vocab]
    return words, non_vocab_words

start_string = input("Please input some text to initiate the lyrics generation (caps insensitive):\n")
start_words, non_vocab_words = clean_and_split_input(start_string)

if non_vocab_words:
    print(f"Words not in the vocabulary: {', '.join(non_vocab_words)}")
else:
    generated_text = generate_text(model, start_string=' '.join(start_words), num_generate=250)
    print("\nGenerated text:\n", generated_text)
