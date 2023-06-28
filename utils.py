import string
from typing import List

def split_text(text: str):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = text.split(" ")
    return text

def get_word_counts(words: List[str], keeptop=None):
    word_counts = {}

    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    if "" in word_counts.keys(): del word_counts[""]

    if keeptop is not None:
        word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:keeptop])
    

    return word_counts

