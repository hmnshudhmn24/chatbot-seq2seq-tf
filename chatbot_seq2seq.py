import tensorflow as tf
import numpy as np

# Sample toy dataset
questions = ["hi", "how are you", "what is your name", "bye"]
answers = ["hello", "I am fine", "I am a chatbot", "goodbye"]

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1

def tokenize_pairs(qs, ans):
    qs_seq = tokenizer.texts_to_sequences(qs)
    ans_seq = tokenizer.texts_to_sequences(ans)
    qs_seq = tf.keras.preprocessing.sequence.pad_sequences(qs_seq, padding='post')
    ans_seq = tf.keras.preprocessing.sequence.pad_sequences(ans_seq, padding='post')
    return qs_seq, ans_seq

encoder_input, decoder_output = tokenize_pairs(questions, answers)
decoder_input = np.insert(decoder_output[:, :-1], 0, 0, axis=1)  # prepend start token (0)

# Build model
embedding_dim = 64
units = 128

# Encoder
encoder_inputs = tf.keras.Input(shape=(None,))
x = tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(units, return_state=True)(x)

# Decoder
decoder_inputs = tf.keras.Input(shape=(None,))
x = tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(x, initial_state=[state_h, state_c])
decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation='softmax')
outputs = decoder_dense(decoder_outputs)

# Model
model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare target for training
decoder_target = np.expand_dims(decoder_output, -1)

# Train
model.fit([encoder_input, decoder_input], decoder_target, batch_size=2, epochs=300, verbose=0)

# Chat function
def chat(input_text):
    seq = tokenizer.texts_to_sequences([input_text])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=encoder_input.shape[1], padding='post')
    enc_out, enc_h, enc_c = model.layers[3](model.layers[2](seq))
    dec_input = np.zeros((1, 1))  # Start token
    decoded = []

    for _ in range(10):
        embed = model.layers[5](dec_input)
        output, state_h, state_c = model.layers[6](embed, initial_state=[enc_h, enc_c])
        preds = model.layers[7](output)
        pred_id = tf.argmax(preds[0, -1, :]).numpy()
        if pred_id == 0:
            break
        decoded.append(tokenizer.index_word.get(pred_id, ''))
        dec_input = np.array([[pred_id]])
        enc_h, enc_c = state_h, state_c
    return ' '.join(decoded)

# Test
print("Bot:", chat("hi"))
print("Bot:", chat("how are you"))
print("Bot:", chat("what is your name"))