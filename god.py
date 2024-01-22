import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import os

# Pfad zum Tokenizer und Modell
script_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(script_dir, "tokenizer.json")
model_path = os.path.join(script_dir, "gru_model.h5")

# Lade den Tokenizer und zeige einige Wörter aus dem Wörterbuch an
with open(tokenizer_path, "r", encoding="utf-8") as file:
    tokenizer = tokenizer_from_json(file.read())

# Modell laden
model = load_model(model_path)


def generate_text(model, tokenizer, seed_text, num_words):
    for _ in range(num_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=model.input_shape[1], padding="post")
        pred_probs = model.predict(encoded)[0]

        # Ignoriere den Padding-Index (0) bei der Vorhersage
        pred_probs[0] = 0
        pred_index = np.argmax(pred_probs)

        pred_word = None
        for word, index in tokenizer.word_index.items():
            if index == pred_index:
                pred_word = word
                break

        if pred_word is None:
            print("Kein Wort für den vorhergesagten Index gefunden. Abbruch.")
            break

        seed_text += " " + pred_word

    return seed_text


# Beispiel: Generiere Text mit dem Modell
seed_text = "Ich  bin"  # Starttext
generated_text = generate_text(model, tokenizer, seed_text, 20)  # Generiere 20 Wörter
print(generated_text)
