# **AI-Powered Quiz Generator**

Transform any block of text into an interactive multiple-choice quiz.

This project was developed as part of the **Fundamentals of AI and ML** course. It demonstrates the use of keyword extraction, semantic similarity, and API-based system design to create an effective learning tool.

---

## **Overview**

Many students struggle to test their understanding while studying. This project addresses that problem by automatically generating quiz questions from any given text. It encourages active recall, which is proven to improve learning and retention.

---

## **Features**

* Automatically extracts important keywords using TF-IDF and RAKE
* Generates multiple-choice questions with meaningful distractors
* Provides instant scoring and performance analysis
* Offers a REST API for easy integration
* Includes a simple browser-based interface

---

## **Project Structure**

```
quiz-generator/
├── data/
├── docs/
├── src/
│   ├── keyword_extractor.py
│   ├── question_generator.py
│   ├── quiz_evaluator.py
│   └── app.py
├── tests/
├── README.md
└── requirements.txt
```

---

## **How to Run**

1. Install dependencies

   ```
   pip install -r requirements.txt
   ```

2. Start the server

   ```
   python src/app.py
   ```

3. Open the frontend

   * Open `docs/index.html` in your browser

---

## **How It Works**

The system follows a simple pipeline:

1. Extracts important keywords from the input text
2. Selects meaningful sentences
3. Converts them into fill-in-the-blank questions
4. Generates plausible incorrect options using semantic similarity
5. Evaluates user answers and provides a performance report

---

## **Technologies Used**

* `rake-nltk` for keyword extraction
* `nltk` for text processing
* `sentence-transformers` for semantic similarity
* `flask` for backend API
* `pytest` for testing

---

## **Conclusion**

This project demonstrates how AI techniques can be applied to improve learning. By automating quiz generation, it helps students actively engage with their study material and assess their understanding more effectively.

---


