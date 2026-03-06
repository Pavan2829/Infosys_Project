import spacy
import json
import os
import spacy
import json
import os

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):

    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    os.makedirs("data/kg", exist_ok=True)

    with open("data/kg/entities.json","w") as f:
        json.dump(entities,f,indent=4)

    print("✅ Entities saved")
    return entities
