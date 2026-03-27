import random
import re
from dataclasses import dataclass, field

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from keyword_extractor import KeywordExtractor

# Optional: sentence transformers
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

# Setup
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))


@dataclass
class QuizQuestion:
    question: str
    options: list[str]
    answer: str
    explanation: str
    difficulty: str


@dataclass
class QuizResult:
    topic: str
    questions: list[QuizQuestion] = field(default_factory=list)


class QuestionGenerator:

    FALLBACK = [
        "algorithm", "model", "data", "function", "variable",
        "matrix", "probability", "feature", "training", "output"
    ]

    def __init__(self, num_questions=5):
        self.num_questions = num_questions
        self.extractor = KeywordExtractor(max_keywords=num_questions + 5)
        self.model = SentenceTransformer("all-MiniLM-L6-v2") if HAS_MODEL else None

    def generate(self, text, topic="General"):
        result = QuizResult(topic=topic)

        keywords = self.extractor.extract(text)
        sentences = sent_tokenize(text)

        for kw in keywords:
            if len(result.questions) >= self.num_questions:
                break

            keyword = kw["keyword"]
            sentence = self._pick_sentence(keyword, sentences)

            if not sentence:
                continue

            question = self._make_question(keyword, sentence, text)
            if question:
                result.questions.append(question)

        return result


    def _pick_sentence(self, keyword, sentences):
        """Pick a decent sentence containing the keyword."""
        for s in sentences:
            if keyword.lower() in s.lower() and len(s.split()) > 6:
                return s
        return None

    def _make_question(self, keyword, sentence, text):
        """Convert sentence into fill-in-the-blank MCQ."""
        blank = re.sub(keyword, "______", sentence, flags=re.IGNORECASE, count=1)

        distractors = self._get_distractors(keyword, text)
        if len(distractors) < 3:
            return None

        options = [keyword] + distractors[:3]
        random.shuffle(options)

        return QuizQuestion(
            question=f"Fill in the blank: {blank}",
            options=options,
            answer=keyword,
            explanation=sentence,
            difficulty=self._difficulty(keyword, sentence)
        )

    def _get_distractors(self, keyword, text):
        """Generate wrong options."""
        words = [
            w for w in word_tokenize(text.lower())
            if w.isalpha() and w not in STOPWORDS and len(w) > 3
        ]

        words = list(set(words))
        words = [w for w in words if keyword.lower() not in w]

        # Use embeddings if available
        if self.model and words:
            try:
                kw_emb = self.model.encode(keyword)
                word_emb = self.model.encode(words)

                scores = util.cos_sim(kw_emb, word_emb)[0].tolist()
                ranked = sorted(zip(words, scores), key=lambda x: abs(x[1] - 0.4))
                return [w for w, _ in ranked[:3]]
            except:
                pass

        # fallback
        random.shuffle(self.FALLBACK)
        return self.FALLBACK[:3]

    def _difficulty(self, keyword, sentence):
        """Simple difficulty logic."""
        if len(sentence.split()) < 10:
            return "easy"
        if len(sentence.split()) > 20 or len(keyword.split()) > 2:
            return "hard"
        return "medium"
