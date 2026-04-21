"""
test_agent.py — Quick test suite for Physics Study Buddy
Run: python test_agent.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("❌ GROQ_API_KEY not set. Add it to .env")
    exit(1)

print("Loading agent (this takes ~30s first run while downloading the embedding model)…")
from agent import get_app, DOCUMENTS  # noqa: E402

app, embedder, collection = get_app()
print(f"✅ Agent ready | KB: {collection.count()} documents\n")


# ── Helper ─────────────────────────────────────────────────────────────────────
def ask(question: str, thread_id: str = "test") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({"question": question}, config=config)


# ── Test questions ─────────────────────────────────────────────────────────────
TESTS = [
    # Domain — expect physics answer
    ("What is Newton's Second Law and how is it applied?",              False),
    ("Derive the range formula for projectile motion.",                  False),
    ("Explain the Work-Energy Theorem with an example.",                 False),
    ("What does the Second Law of Thermodynamics state?",               False),
    ("State Coulomb's Law and explain the electric field concept.",      False),
    ("How does electromagnetic induction work? Give Faraday's Law.",    False),
    ("What is the Doppler Effect and when does it occur?",              False),
    ("What is the period formula for a simple pendulum?",               False),
    # Red-team
    ("What is the best recipe for chocolate cake?",                     True),
    ("Einstein proved that nothing can travel faster than light, so photons have mass, right?", True),
]

results = []
print("=" * 65)
print("TEST SUITE — Physics Study Buddy")
print("=" * 65)

for i, (q, red_team) in enumerate(TESTS):
    label = "[RED TEAM]" if red_team else ""
    print(f"\n--- Test {i+1} {label} ---")
    print(f"Q: {q}")

    result  = ask(q, thread_id=f"test-{i}")
    answer  = result.get("answer", "")
    faith   = result.get("faithfulness", 0.0)
    route   = result.get("route", "?")

    print(f"A: {answer[:280]}")
    print(f"Route: {route} | Faithfulness: {faith:.2f}")

    # Pass criteria
    if red_team and i == 8:  # out-of-scope food question
        passed = any(kw in answer.lower() for kw in [
            "not a physics", "outside", "don't know", "cannot help",
            "not related", "scope", "not about physics"
        ])
    elif red_team and i == 9:  # false-premise photon question
        passed = any(kw in answer.lower() for kw in [
            "massless", "no mass", "photons do not", "incorrect",
            "not correct", "actually", "photons are"
        ])
    else:
        passed = len(answer) > 50 and any(kw in answer.lower() for kw in [
            "force", "energy", "velocity", "mass", "motion", "law",
            "equation", "field", "wave", "quantum", "acceleration",
            "temperature", "electric", "magnetic", "doppler",
            "newton", "einstein", "faraday", "period", "pendulum",
        ])

    print(f"Result: {'✅ PASS' if passed else '❌ FAIL'}")
    results.append(passed)

total  = len(results)
passed = sum(results)
print(f"\n{'='*65}")
print(f"RESULTS: {passed}/{total} passed")
avg_faith = 0.0  # faithfulness already printed per test
print("Run complete.")
