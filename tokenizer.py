import spacy
import os

# Lade das deutsche NLP-Modell
nlp = spacy.load("de_core_news_sm")


def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "cleaned_bible_text.txt")

# Lese den Text aus der Datei
with open(file_path, "r", encoding="windows-1252") as file:
    text = file.read()

# Aufteilen des Textes in Abschnitte
chunk_size = 100000
text_chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

tokenized_text = []

# Verarbeite jeden Abschnitt einzeln
for i, chunk in enumerate(text_chunks, 1):
    tokens = tokenize_text(chunk)
    tokenized_text.extend(tokens)
    print(f"Abschnitt {i} von {len(text_chunks)} verarbeitet.")

# Speichere die tokenisierten Daten
output_path = os.path.join(script_dir, "tokenized_text.txt")
with open(output_path, "w", encoding="utf-8") as file:
    for token in tokenized_text:
        file.write(token + "\n")

print("Tokenisierung abgeschlossen und gespeichert.")
