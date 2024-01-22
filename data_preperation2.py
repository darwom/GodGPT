import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "cleaned_bible_text.txt")

# Lese den Text aus der Datei
with open(file_path, "r", encoding="windows-1252") as file:
    text = file.read()

# Teile den Text in Sätze
sentences = sent_tokenize(text, language="german")

# Erstelle und trainiere den Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Konvertiere jeden Satz in eine Sequenz
sequences = tokenizer.texts_to_sequences(sentences)

# Bestimme eine geeignete Sequenzlänge
max_seq_length = 50  # Beispielwert, anpassen nach Bedarf

# Führe Padding für jede Sequenz durch
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding="post")

# Aufteilen in Trainings- und Testdaten
split_frac = 0.8
split_idx = int(len(padded_sequences) * split_frac)
train_seq = padded_sequences[:split_idx]
test_seq = padded_sequences[split_idx:]

print(f"Trainingsdaten: {len(train_seq)}")
print(f"Testdaten: {len(test_seq)}")

# Speichere Trainings- und Testdaten
np.save(os.path.join(script_dir, "train_seq.npy"), train_seq)
np.save(os.path.join(script_dir, "test_seq.npy"), test_seq)

# Speichere den Tokenizer
tokenizer_json = tokenizer.to_json()
with open(os.path.join(script_dir, "tokenizer.json"), "w", encoding="utf-8") as file:
    file.write(tokenizer_json)
