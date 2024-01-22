import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.callbacks import EarlyStopping
import os

# Pfad einstellen, um Trainings- und Testdaten sowie den Tokenizer zu laden
script_dir = os.path.dirname(os.path.abspath(__file__))
train_seq_path = os.path.join(script_dir, "train_seq.npy")
test_seq_path = os.path.join(script_dir, "test_seq.npy")
tokenizer_path = os.path.join(script_dir, "tokenizer.json")

# Tokenizer und Daten laden
with open(tokenizer_path, "r", encoding="utf-8") as file:
    tokenizer = tokenizer_from_json(file.read())
train_seq = np.load(train_seq_path)
test_seq = np.load(test_seq_path)

# Parameter definieren
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
gru_units = 256
dropout_rate = 0.6
l2_lambda = 0.001

# Modellarchitektur
model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=train_seq.shape[1] - 1,
    )
)
model.add(GRU(gru_units, return_sequences=True, kernel_regularizer=l2(l2_lambda)))
model.add(Dropout(dropout_rate))
model.add(GRU(gru_units, kernel_regularizer=l2(l2_lambda)))
model.add(Dropout(dropout_rate))
model.add(Dense(vocab_size, activation="softmax"))

# Modell kompilieren
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Trainings- und Testdaten vorbereiten
X_train = train_seq[:, :-1]
y_train = train_seq[:, -1]
X_test = test_seq[:, :-1]
y_test = test_seq[:, -1]

# EarlyStopping
early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1, mode="min")

# Modell trainieren
model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
)

# Modell speichern
model.save("gru_model_enhanced_v3.h5")
