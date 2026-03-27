from flask import Flask, request, jsonify
from flask_cors import CORS

from question_generator import QuestionGenerator, QuizQuestion
from quiz_evaluator import QuizEvaluator

app = Flask(__name__)
CORS(app)

generator = QuestionGenerator(num_questions=10)
evaluator = QuizEvaluator()

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()

    text = data.get("text", "").strip()
    topic = data.get("topic", "General")
    num_q = int(data.get("num_questions", 10))

    if not text or len(text.split()) < 30:
        return jsonify({"error": "Provide at least 30 words of text"}), 400

    generator.num_questions = max(3, min(num_q, 20))

    result = generator.generate(text, topic)

    questions = [
        {
            "id": i,
            "question": q.question,
            "options": q.options,
            "answer": q.answer,
            "difficulty": q.difficulty
        }
        for i, q in enumerate(result.questions)
    ]

    return jsonify({
        "topic": result.topic,
        "total": len(questions),
        "questions": questions
    })

@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()

    q_data = data.get("questions", [])
    answers = data.get("user_answers", [])

    if not q_data or not answers:
        return jsonify({"error": "Questions and answers required"}), 400

    questions = [
        QuizQuestion(
            question=q["question"],
            options=q["options"],
            answer=q["answer"],
            explanation=q.get("explanation", ""),
            difficulty=q.get("difficulty", "medium")
        )
        for q in q_data
    ]

    report = evaluator.evaluate(questions, answers)

    return jsonify({
        "score": report.score,
        "grade": report.grade,
        "performance": report.performance,
        "correct": report.correct,
        "total": report.total,
        "difficulty": report.difficulty_stats
    })

if __name__ == "__main__":
    app.run(debug=True)
