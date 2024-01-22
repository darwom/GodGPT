import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import random
import os


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, tokenizer, seed_text, num_words, temperature=1.0):
    for _ in range(num_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=model.input_shape[1], padding="post")
        pred_probs = model.predict(encoded)[0]

        # Ignoriere den Index 0 (Padding)
        pred_probs[0] = 0
        pred_index = sample(pred_probs, temperature)

        pred_word = None
        for word, index in tokenizer.word_index.items():
            if index == pred_index:
                pred_word = word
                break

        if pred_word is None:
            print("Kein weiteres Wort gefunden. Abbruch.")
            break

        seed_text += " " + pred_word

    return seed_text


# Modell und Tokenizer laden
script_dir = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(script_dir, "gru_model_enhanced.h5"))

tokenizer_path = os.path.join(script_dir, "tokenizer.json")
with open(tokenizer_path, "r", encoding="utf-8") as file:
    tokenizer = tokenizer_from_json(file.read())

# Text generieren
seed_text = "Am Anfang schuf Gott"
generated_text = generate_text(model, tokenizer, seed_text, 20, temperature=0.5)
print(generated_text)
