try:
    import spacy  # type: ignore
except Exception:
    spacy = None
import json
import os
import re


# Try to load spaCy model; if unavailable, set nlp=None so caller can fallback
if spacy is not None:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
else:
    nlp = None


def clean_relation(rel: str) -> str | None:
    """Normalize relation string to uppercase letters and underscores.

    Returns None if the cleaned relation is too short.
    """
    if not rel:
        return None
    rel = rel.upper()
    rel = re.sub(r'[^A-Z_]', '', rel)
    if len(rel) < 3:
        return None
    return rel


def _get_span_text(token):
    """Return the full text span for a token (including compounds and modifiers)."""
    parts = [t.text for t in token.subtree]
    s = " ".join(parts).strip()
    # Normalize whitespace and remove newlines/hard hyphens
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'\s+', ' ', s)
    # Remove trailing punctuation and hyphens
    s = s.strip().strip(' ,.;:-')
    return s


def extract_triplets(text: str):
    """Extract (subject, relation, object) triplets from text using spaCy dependencies.

    This function returns a list of tuples and does NOT write files to disk
    (saving is handled by the caller).
    """
    # If spaCy is not available, return empty list rather than raising an ImportError
    if nlp is None:
        return []

    doc = nlp(text)
    triplets = []

    for sent in doc.sents:
        for token in sent:
            # Consider ROOT verbs (dependency label can vary in case)
            if token.dep_.lower() == "root":
                subject = None
                obj = None

                for child in token.children:
                    dep = child.dep_.lower()
                    if dep in ("nsubj", "nsubjpass", "csubj", "agent"):
                        subject = _get_span_text(child)
                    if dep in ("dobj", "pobj", "dative", "attr", "oprd", "obj"):
                        obj = _get_span_text(child)

                # Clean and filter spans
                if subject and obj:
                    # Normalize
                    subject = subject.strip()
                    obj = obj.strip()

                    # Filter out pronouns and very short spans
                    low_sub = subject.lower()
                    first_word = low_sub.split()[0]
                    if first_word in {"we","i","you","they","he","she","it","our","their","us","them","this","that","these","those"}:
                        continue
                    if len(subject) < 2 or len(obj) < 2:
                        continue

                    relation = clean_relation(token.lemma_)
                    if relation:
                        triplets.append((subject, relation, obj))

    return triplets
