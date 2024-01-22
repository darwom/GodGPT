import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Setze den Pfad für deine tokenisierte Textdatei
script_dir = os.path.dirname(os.path.abspath(__file__))
tokenized_file_path = os.path.join(script_dir, "tokenized_text.txt")

# Lese die tokenisierten Daten
with open(tokenized_file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Aufteilen der Daten in Abschnitte
chunk_size = 10000  # Anzahl der Zeilen pro Abschnitt
chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

# Erstelle ein Tokenizer-Objekt
tokenizer = Tokenizer()
for chunk in chunks:
    tokenizer.fit_on_texts(chunk)

# Konvertiere alle Daten in numerische Sequenzen
sequences = tokenizer.texts_to_sequences(lines)

# Bestimme die maximale Sequenzlänge
max_seq_length = max([len(seq) for seq in sequences])

# Führe Padding durch
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding="post")

# Teile die Daten in Trainings- und Testsets
split_frac = 0.8
split_idx = int(len(padded_sequences) * split_frac)
train_seq, test_seq = padded_sequences[:split_idx], padded_sequences[split_idx:]

# Deine Daten sind jetzt bereit für das Training mit einem GRU-Modell

# Speichere die Trainings- und Testdaten
np.save(os.path.join(script_dir, "train_seq.npy"), train_seq)
np.save(os.path.join(script_dir, "test_seq.npy"), test_seq)
