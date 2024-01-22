import xml.etree.ElementTree as ET
import os


def clean_bible_text(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Definiere den Namespace
    ns = {"osis": "http://www.bibletechnologies.net/2003/OSIS/namespace"}

    clean_text = ""
    for verse in root.findall(".//osis:verse", ns):
        if verse.text:
            clean_text += verse.text + " "

    return clean_text


# Pfad zu dem Verzeichnis, in dem sich dieses Skript befindet
script_dir = os.path.dirname(os.path.abspath(__file__))

# Pfad zu der Datei 'luth1912ap.xml' im selben Verzeichnis
file_path = os.path.join(script_dir, "luth1912ap.xml")

# Verwende diese Funktion mit deinem XML-Dateipfad
cleaned_text = clean_bible_text(file_path)

# Speichere den bereinigten Text in einer neuen Datei
with open("cleaned_bible_text.txt", "w") as file:
    file.write(cleaned_text)

print("Done!")
