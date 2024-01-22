import spacy

# Lemmatisierung also die Reduktion von Wortformen auf ihre Grundform
# ! Machen wir aber nicht !
# Wie ich es verstehe würde das Modell dadurch nie die verschiedenen Formen lernen
"""
# Lade das deutsche NLP-Modell
nlp = spacy.load("de_core_news_sm")


def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text


# Beispieltext
text = "Am Anfang schuf Gott Himmel und Erde. Und die Erde war wüst und leer,"

# Lemmatisierung durchführen
lemmatized_text = lemmatize_text(text)

# Speichern des lemmatisierten Textes
with open("lemmatized_text.txt", "w", encoding="utf-8") as file:
    file.write(lemmatized_text)

print("Lemmatisierung abgeschlossen und gespeichert.")
"""
