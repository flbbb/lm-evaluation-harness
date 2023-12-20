import re

from nltk.tokenize import sent_tokenize


def wikitext_detokenizer(doc):
    string = doc["page"]
    # contractions
    string = string.replace("s '", "s'")
    # string = " ".join([s.capitalize() for s in sent_tokenize(string)])
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = string.replace(" -rrb- ", ") ")
    string = string.replace(" -lrb- ", " (")
    string = string.replace("`` ", '"')
    string = string.replace(" ''", '"')
    # miscellaneous (most are from wikitext)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def process_results(doc, results):
    (loglikelihood,) = results
    # IMPORTANT: wikitext counts number of words in *original doc before detokenization*
    _words = len(re.split(r"\s+", doc["page"]))
    _bytes = len(doc["page"].encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }
