import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Pfad einstellen, um Trainings- und Testdaten sowie den Tokenizer zu laden
script_dir = os.path.dirname(os.path.abspath(__file__))
train_seq_path = os.path.join(script_dir, "train_seq.npy")
test_seq_path = os.path.join(script_dir, "test_seq.npy")
tokenizer_path = os.path.join(script_dir, "tokenizer.json")

# Trainings- und Testdaten laden
train_seq = np.load(train_seq_path)
test_seq = np.load(test_seq_path)

# Tokenizer laden
with open(tokenizer_path, "r", encoding="utf-8") as file:
    tokenizer = tokenizer_from_json(file.read())

# Modellparameter definieren
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
gru_units = 128

# Modellarchitektur
model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=train_seq.shape[1] - 1,
    )
)
model.add(GRU(gru_units, return_sequences=True))
model.add(GRU(gru_units))
model.add(Dense(vocab_size, activation="softmax"))

# Modell kompilieren
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Trainingsdaten vorbereiten
X_train = train_seq[:, :-1]
y_train = train_seq[:, -1]

# Testdaten vorbereiten
X_test = test_seq[:, :-1]
y_test = test_seq[:, -1]

# Modell trainieren
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Modell speichern
model.save("gru_model.h5")
