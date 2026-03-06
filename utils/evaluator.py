import re
from collections import Counter

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except:
    HAS_BERTSCORE = False


# -----------------------------
# NGRAM GENERATOR
# -----------------------------
def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


# -----------------------------
# ROUGE-1 / ROUGE-2
# -----------------------------
def _score_ngrams(generated, reference, n):

    gen_tokens = re.findall(r"\w+", generated.lower())
    ref_tokens = re.findall(r"\w+", reference.lower())

    if not gen_tokens or not ref_tokens:
        return {'precision':0.0,'recall':0.0,'f1':0.0}

    gen_ngrams = Counter(_ngrams(gen_tokens, n))
    ref_ngrams = Counter(_ngrams(ref_tokens, n))

    overlap = sum((gen_ngrams & ref_ngrams).values())

    precision = overlap / sum(gen_ngrams.values()) if gen_ngrams else 0.0
    recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0

    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

    return {'precision':precision,'recall':recall,'f1':f1}


# -----------------------------
# FAST ROUGE-L (NO DP MATRIX)
# -----------------------------
def _rouge_l(generated, reference):

    gen = re.findall(r"\w+", generated.lower())
    ref = re.findall(r"\w+", reference.lower())

    if not gen or not ref:
        return 0.0

    i = j = lcs = 0

    while i < len(gen) and j < len(ref):
        if gen[i] == ref[j]:
            lcs += 1
            i += 1
            j += 1
        else:
            j += 1

    return lcs / len(ref)


# -----------------------------
# FINAL EVALUATION
# -----------------------------
def evaluate_llm(generated, reference):

    r1 = _score_ngrams(generated, reference, 1)
    r2 = _score_ngrams(generated, reference, 2)
    rl = _rouge_l(generated, reference)

    result = {
        'rouge1': r1,
        'rouge2': r2,
        'rougeL': {'lcs_ratio': rl}
    }

    # -------- BERTScore --------
    if HAS_BERTSCORE:
        try:
            P, R, F1 = bert_score([generated], [reference], lang="en")

            result['bertscore'] = {
                'precision': P[0].item(),
                'recall': R[0].item(),
                'f1': F1[0].item()
            }
        except:
            pass

    return result