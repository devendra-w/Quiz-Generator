import re
import math
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from rake_nltk import Rake

# Download required NLTK data on first use
def _ensure_nltk_data():
    for resource in ["punkt", "stopwords", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()


class KeywordExtractor:
    """
    Combines RAKE and TF-IDF to extract the most meaningful keywords
    and key phrases from a given passage of text.
    """

    def __init__(self, max_keywords: int = 15):
        self.max_keywords = max_keywords
        self._stop_words = set(stopwords.words("english"))
        self._rake = Rake()

    

    def extract(self, text: str) -> list[dict]:
        """
        Return a ranked list of keyword dicts.

        Each dict has:
            keyword (str)  – the word or phrase
            score   (float)– combined relevance score (0–1 normalised)
        """
        text = self._clean(text)
        rake_phrases = self._rake_extract(text)
        tfidf_words  = self._tfidf_extract(text)

        combined = self._merge(rake_phrases, tfidf_words)
        return combined[: self.max_keywords]

   

    def _clean(self, text: str) -> str:
        """Remove special characters and normalise whitespace."""
        text = re.sub(r"[^\w\s\.\,\!\?]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _rake_extract(self, text: str) -> list[tuple[str, float]]:
        """Run RAKE and return (phrase, score) pairs."""
        self._rake.extract_keywords_from_text(text)
        ranked = self._rake.get_ranked_phrases_with_scores()  # [(score, phrase)]
        if not ranked:
            return []

        max_score = max(score for score, _ in ranked) or 1.0
        return [(phrase, score / max_score) for score, phrase in ranked[:20]]

    def _tfidf_extract(self, text: str) -> list[tuple[str, float]]:
        """
        Compute a simple single-document TF-IDF (using sentence-level IDF)
        and return (word, score) pairs for content words.
        """
        sentences = sent_tokenize(text)
        n = len(sentences) or 1

        # Term frequency across whole document
        tokens = [t.lower() for t in word_tokenize(text)
                  if t.isalpha() and t.lower() not in self._stop_words]
        tf = Counter(tokens)
        total = sum(tf.values()) or 1

        # IDF: how many sentences contain each word
        idf: dict[str, float] = {}
        for word in tf:
            df = sum(1 for s in sentences if word in s.lower())
            idf[word] = math.log((n + 1) / (df + 1)) + 1

        max_score = max((tf[w] / total * idf[w]) for w in tf) if tf else 1.0
        results = [(w, (tf[w] / total * idf[w]) / max_score) for w in tf]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:20]

    def _merge(
        self,
        rake_phrases: list[tuple[str, float]],
        tfidf_words:  list[tuple[str, float]],
    ) -> list[dict]:
        """
        Combine scores from both methods.
        Phrases from RAKE carry extra weight (×1.5) because multi-word
        concepts are usually more question-worthy.
        """
        scores: dict[str, float] = {}

        for phrase, score in rake_phrases:
            scores[phrase] = scores.get(phrase, 0.0) + score * 1.5

        for word, score in tfidf_words:
            # Don't add a single word if it's already part of a phrase
            already_covered = any(word in phrase for phrase in scores)
            if not already_covered:
                scores[word] = scores.get(word, 0.0) + score

        # Normalise to [0, 1]
        if scores:
            max_s = max(scores.values())
            scores = {k: v / max_s for k, v in scores.items()}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"keyword": kw, "score": round(sc, 4)} for kw, sc in ranked]
