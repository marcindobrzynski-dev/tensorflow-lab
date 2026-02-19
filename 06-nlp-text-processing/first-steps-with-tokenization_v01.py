import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'Dzisiaj mamy słoneczny dzień',
    'Dzisiaj mamy deszczowy dzień',
    'Czy dzisiaj mamy słoneczny dzień?',
    'Bardzo podobał mi się dzisiejszy spacer po śniegu'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
print('\n')

padded = pad_sequences(sequences, padding='post', maxlen=6, truncating='post')
print(padded)