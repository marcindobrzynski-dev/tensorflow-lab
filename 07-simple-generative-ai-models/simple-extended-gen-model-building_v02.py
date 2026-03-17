!wget --no-check-certificate \
    https://storage.googleapis.com/tensorflow-1-public/course3/irish-lyrics-eof.txt \
    -O /tmp/irish-lyrics-eof.txt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

import matplotlib.pyplot as plt
import numpy as np

tokenizer = Tokenizer()

# Prepare data
window_size = 11
max_sequence_len = 64

sentences = []
all_text = []
data = open("/tmp/irish-lyrics-eof.txt").read()
corpus = data.lower()
words = corpus.split(" ")
range_size = len(words) - window_size

for i in range(0, range_size + 1):
  this_sentence = ""

  for word in range(0, window_size):
    word = words[i + word]
    this_sentence += word + " "

  sentences.append(this_sentence)

# Transform data to tokens
tokenizer.fit_on_texts(sentences)

# Count number of tokens in tokenizer
total_words = len(tokenizer.word_index) + 1

# Prepare inputs and labels
xs = []
labels = []

for line in sentences:
  token_list = tokenizer.texts_to_sequences([line])[0]

  if (len(token_list) == window_size):
    xs.append(token_list[:window_size - 1])
    labels.append(token_list[window_size - 1])

xs = np.array(xs)

ys = tf.keras.utils.to_categorical(
    labels,
    num_classes=total_words
)

# Model training
model = Sequential()
model.add(Embedding(total_words, 64))
model.add(Bidirectional(LSTM(max_sequence_len, dropout=0.2)))
model.add(Dense(total_words, activation='softmax'))

steps_per_epoch = len(xs) // 32
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=steps_per_epoch * 50,
    decay_rate=0.8,
    staircase=True
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=50,
    min_delta=0.05,
    restore_best_weights=True
)

history = model.fit(
    xs,
    ys,
    epochs=500,
    callbacks=[early_stop],
    verbose=1
)

# Show model training specifications
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.show()

plt.ylabel("Accuracy")
plot_graphs(history, 'accuracy')
plt.ylabel("Loss")
plot_graphs(history, 'loss')

## First generation of text
seed_text = "Many years ago"
next_words = 25

for _ in range(next_words):
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences(
    [token_list],
    maxlen=window_size - 1,
    padding='pre'
  )

  predictions = model.predict(token_list, verbose=0)[0]

  temperature = 0.8
  predictions = np.log(predictions + 1e-8) / temperature
  exp_preds = np.exp(predictions)
  predictions = exp_preds / np.sum(exp_preds)

  recent_tokens = token_list[0][-3:]

  for token in recent_tokens:
    if token > 0:
      predictions[token] *= 0.3

  predictions = predictions / np.sum(predictions)

  predicted = np.random.choice(total_words, p=predictions)

  output_word = ""

  for word, index in tokenizer.word_index.items():
    if index == predicted:
      output_word = word
      break

  seed_text += " " + output_word

print(seed_text)
