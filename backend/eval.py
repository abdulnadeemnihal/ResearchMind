"""Retrieval check for ResearchMind AI. Uses only the Python standard library.

Start the backend and upload a PDF, then run:
  python eval.py
  python eval.py http://localhost:8000
"""
import json
import sys
import urllib.request

URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

OFF_TOPIC_QUESTIONS = [
    "Who won the world cup 2025?",
    "Who won?",
    "What is the capital of France?",
]


def get_suggested_questions():
    req = urllib.request.Request(
        URL + "/suggested-questions",
        data=json.dumps({"doc_id": None}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)["questions"]


def search(question):
    req = urllib.request.Request(
        URL + "/search",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def main():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=10) as response:
            stored = json.load(response)["chunks_stored"]
    except Exception as error:
        sys.exit(f"Cannot reach the backend at {URL}: {error}")
    if stored == 0:
        sys.exit("The database is empty. Upload a PDF first.")
    print(f"Backend OK, {stored} chunks stored.\n")

    try:
        suggestions = get_suggested_questions()
    except Exception as error:
        sys.exit(f"Could not generate questions: {error}")

    questions = [(item["question"], True) for item in suggestions]
    questions.extend((question, False) for question in OFF_TOPIC_QUESTIONS)
    print("Suggested questions for the uploaded PDF:")
    for item in suggestions:
        print(f"- {item['question']}")
    print()

    good_best = []
    bad_best = []
    hits = 0
    print(f"{'best':>6}  {'result':<12} {'pages in top 4':<18} question")
    for question, expected in questions:
        results = search(question)
        best = min(result["distance"] for result in results)
        pages = sorted({result["page"] for result in results})
        if not expected:
            bad_best.append(best)
            verdict = "off-topic"
        else:
            good_best.append(best)
            found = bool(pages)
            hits += found
            verdict = "FOUND" if found else "MISSED"
        print(f"{best:6.3f}  {verdict:<12} {str(pages):<18} {question}")

    on_topic = len(good_best)
    print(f"\nRight page in top 4: {hits}/{on_topic} on-topic questions")
    if good_best and bad_best:
        worst_good = max(good_best)
        best_bad = min(bad_best)
        print(f"Worst on-topic best-distance: {worst_good:.3f}")
        print(f"Closest off-topic distance:   {best_bad:.3f}")
        if worst_good < best_bad:
            midpoint = (worst_good + best_bad) / 2
            print(f"\nSuggested MAX_DISTANCE = {midpoint:.2f} (set it at the top of main.py)")
        else:
            print("\nThe ranges overlap: no single MAX_DISTANCE separates them cleanly.")
            print("Send these results to your assistant before changing the value.")


if __name__ == "__main__":
    main()
