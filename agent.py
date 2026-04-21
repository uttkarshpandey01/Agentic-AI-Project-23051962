"""
agent.py — Physics Study Buddy shared backend
Exports: get_app(), get_embedder(), get_collection(), get_llm(), DOCUMENTS

Usage:
    from agent import get_app, DOCUMENTS
    app, embedder, collection = get_app()
"""

import os
from typing import List, TypedDict

import chromadb
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Knowledge Base ─────────────────────────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Newton's Laws of Motion",
        "text": (
            "Newton's three laws of motion form the foundation of classical mechanics. "
            "The First Law (Law of Inertia) states that an object at rest stays at rest, and an object "
            "in motion stays in motion unless acted upon by an unbalanced external force. "
            "The Second Law states that the net force equals mass times acceleration: F = ma. "
            "The Third Law states that for every action there is an equal and opposite reaction. "
            "Applications include engineering design, orbital mechanics, sports science, and everyday "
            "problems involving forces, friction, tension, and acceleration."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Kinematics and Projectile Motion",
        "text": (
            "Kinematics describes motion without considering its causes. Key equations for constant "
            "acceleration: v = u + at, s = ut + ½at², v² = u² + 2as. "
            "Projectile Motion: launched at angle θ with speed v₀, horizontal: x = v₀cosθ·t, "
            "vertical: y = v₀sinθ·t − ½gt². Time of flight T = 2v₀sinθ/g, "
            "range R = v₀²sin2θ/g, max height H = v₀²sin²θ/(2g). Maximum range at θ = 45°."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Work, Energy, and Power",
        "text": (
            "Work: W = F·d·cosθ (joules). Kinetic Energy: KE = ½mv². "
            "Work-Energy Theorem: net work = ΔKE. Gravitational PE = mgh. "
            "Spring PE = ½kx². Conservation of energy: KE₁ + PE₁ = KE₂ + PE₂ (no friction). "
            "Power P = W/t = F·v (watts). Efficiency η = (useful output / total input) × 100%."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Thermodynamics and Heat Transfer",
        "text": (
            "Four laws of thermodynamics. Zeroth: defines temperature via thermal equilibrium. "
            "First: ΔU = Q − W (energy conservation). Second: entropy always increases; "
            "no engine is 100% efficient. Third: entropy → minimum as T → 0 K. "
            "Heat transfer: Conduction Q/t = kAΔT/d; Convection via fluid motion; "
            "Radiation P = εσAT⁴. Ideal Gas Law: PV = nRT."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Electric Fields and Coulomb's Law",
        "text": (
            "Coulomb's Law: F = kq₁q₂/r², k = 8.99×10⁹ N·m²/C². "
            "Electric field E = F/q = kQ/r². Electric potential V = kQ/r. "
            "Gauss's Law: Φ = Q_enc/ε₀. "
            "Capacitance C = Q/V; parallel plate C = ε₀A/d; stored energy U = ½CV²."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Magnetism and Electromagnetic Induction",
        "text": (
            "Magnetic force on a charge: F = qv×B. On a wire: F = IL×B. "
            "Long straight wire: B = μ₀I/(2πr). Solenoid: B = μ₀nI. "
            "Faraday's Law: ε = −dΦ_B/dt. Lenz's Law: induced current opposes change. "
            "Maxwell's equations unify electricity and magnetism; EM waves travel at c = 3×10⁸ m/s."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Waves and Sound",
        "text": (
            "Wave properties: wavelength λ, frequency f, period T = 1/f, speed v = fλ, amplitude A. "
            "Transverse: oscillation ⊥ propagation (light). Longitudinal: oscillation ∥ propagation (sound). "
            "Sound speed ≈ 343 m/s at 20°C. Intensity I = P/(4πr²). dB = 10·log(I/I₀). "
            "Doppler: f_obs = f_src·(v ± v_obs)/(v ∓ v_src). "
            "Standing waves on string: f_n = nv/(2L)."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Optics — Reflection, Refraction, and Lenses",
        "text": (
            "Reflection: angle of incidence = angle of reflection. "
            "Snell's Law: n₁sinθ₁ = n₂sinθ₂. Total internal reflection when θ₁ > θ_c = arcsin(n₂/n₁). "
            "Thin lens: 1/f = 1/d_o + 1/d_i; magnification m = −d_i/d_o. "
            "Converging (convex) f > 0; diverging (concave) f < 0. "
            "Double-slit maxima: d·sinθ = mλ."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Modern Physics — Quantum Mechanics and Special Relativity",
        "text": (
            "Special Relativity: time dilation Δt = γΔt₀; length contraction L = L₀/γ; E = mc²; "
            "γ = 1/√(1−v²/c²). "
            "Quantum Mechanics: E = hf (h = 6.626×10⁻³⁴ J·s). "
            "Photoelectric: KE_max = hf − φ. de Broglie: λ = h/p. "
            "Uncertainty: ΔxΔp ≥ ħ/2. Bohr hydrogen: E_n = −13.6 eV/n²."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Circular Motion and Gravitation",
        "text": (
            "Centripetal acceleration: a_c = v²/r = ω²r. Centripetal force: F_c = mv²/r. "
            "Torque: τ = rFsinθ = Iα. "
            "Newton's gravitation: F = Gm₁m₂/r², G = 6.674×10⁻¹¹ N·m²/kg². "
            "Orbital velocity: v = √(GM/r). Kepler's Third Law: T² = (4π²/GM)r³. "
            "Escape velocity: v_esc = √(2GM/R) ≈ 11.2 km/s for Earth."
        ),
    },
    {
        "id": "doc_011",
        "topic": "Simple Harmonic Motion and Oscillations",
        "text": (
            "SHM restoring force: F = −kx. Displacement: x(t) = A·cos(ωt + φ). "
            "Spring-mass: ω = √(k/m), T = 2π√(m/k). "
            "Simple pendulum: T = 2π√(L/g) (small angles). "
            "Energy: E = ½kA² = constant; all KE at equilibrium, all PE at amplitude. "
            "Resonance: maximum amplitude when driving frequency = natural frequency."
        ),
    },
    {
        "id": "doc_012",
        "topic": "Fluid Mechanics",
        "text": (
            "Pressure P = F/A. Hydrostatic: P = P₀ + ρgh. "
            "Archimedes: F_b = ρ_fluid·V_submerged·g (objects float if avg density < fluid). "
            "Continuity: A₁v₁ = A₂v₂. "
            "Bernoulli: P + ½ρv² + ρgh = constant (faster flow → lower pressure). "
            "Applications: aircraft lift, hydraulic systems, blood flow."
        ),
    },
]

# ── State ──────────────────────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question:       str
    messages:       List[dict]
    route:          str
    retrieved:      str
    sources:        List[str]
    tool_result:    str
    answer:         str
    faithfulness:   float
    eval_retries:   int
    search_results: str


# ── Constants ──────────────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.6
MAX_EVAL_RETRIES       = 2


# ── Factory functions ──────────────────────────────────────────────────────────
def get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_collection(embedder: SentenceTransformer = None) -> chromadb.Collection:
    if embedder is None:
        embedder = get_embedder()
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("physics_kb")
    except Exception:
        pass
    col = client.create_collection("physics_kb")
    texts = [d["text"] for d in DOCUMENTS]
    col.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )
    return col


def get_app():
    """
    Build and return (compiled_graph, embedder, collection).
    Call once per process; cache the result.
    """
    llm        = get_llm()
    embedder   = get_embedder()
    collection = get_collection(embedder)

    # ── Node definitions ───────────────────────────────────────────────────────

    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
        return {"messages": msgs[-6:]}  # sliding window: last 3 turns

    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        recent   = "; ".join(
            f"{m['role']}: {m['content'][:60]}"
            for m in state.get("messages", [])[-3:-1]
        ) or "none"
        prompt = (
            "You are a router for a Physics Study Buddy chatbot.\n"
            "Reply with ONLY one word: retrieve / memory_only / tool\n\n"
            "- retrieve: physics concepts, formulas, or explanations in the KB\n"
            "- memory_only: references to earlier in this conversation\n"
            "- tool: recent research or topics not in the knowledge base\n\n"
            f"Recent: {recent}\nQuestion: {question}"
        )
        raw = llm.invoke(prompt).content.strip().lower()
        if "tool"   in raw: return {"route": "tool"}
        if "memory" in raw: return {"route": "memory_only"}
        return {"route": "retrieve"}

    def retrieval_node(state: CapstoneState) -> dict:
        q_emb   = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks  = results["documents"][0]
        topics  = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    def tool_node(state: CapstoneState) -> dict:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(f"physics {state['question']}", max_results=3))
            if results:
                snippets = "\n\n".join(
                    f"[{r.get('title', 'Source')}]\n{r.get('body', '')}"
                    for r in results
                )
                output = f"Web search results:\n\n{snippets}"
            else:
                output = "No web search results found."
        except Exception as e:
            output = f"Web search unavailable: {e}"
        return {"tool_result": output, "search_results": output}

    def answer_node(state: CapstoneState) -> dict:
        context_parts = []
        if state.get("retrieved"):
            context_parts.append(f"KNOWLEDGE BASE:\n{state['retrieved']}")
        if state.get("tool_result"):
            context_parts.append(f"WEB SEARCH:\n{state['tool_result']}")
        context = "\n\n".join(context_parts)

        if context:
            retry_note = (
                "\n\nIMPORTANT: Previous answer failed quality check. "
                "Be strictly faithful to the context below.\n"
                if state.get("eval_retries", 0) > 0 else ""
            )
            system_content = (
                "You are an expert Physics Study Buddy for college students. "
                "Explain concepts clearly, show formulas with units, work through problems step-by-step, "
                "and connect ideas to real-world examples.\n"
                "Answer using ONLY the information in the context below. "
                "If the answer is not in the context, say so clearly."
                f"{retry_note}\n\n{context}"
            )
        else:
            system_content = (
                "You are a helpful physics assistant. "
                "Answer based on the conversation history."
            )

        lc_msgs = [SystemMessage(content=system_content)]
        for m in state.get("messages", [])[:-1]:
            cls = HumanMessage if m["role"] == "user" else AIMessage
            lc_msgs.append(cls(content=m["content"]))
        lc_msgs.append(HumanMessage(content=state["question"]))

        return {"answer": llm.invoke(lc_msgs).content}

    def eval_node(state: CapstoneState) -> dict:
        retries = state.get("eval_retries", 0)
        context = state.get("retrieved", "")[:500]
        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}
        prompt = (
            "Rate faithfulness 0.0-1.0. Reply with ONLY a number.\n"
            "1.0 = fully faithful to context. 0.0 = mostly hallucinated.\n\n"
            f"Context: {context}\nAnswer: {state.get('answer', '')[:300]}"
        )
        try:
            score = float(llm.invoke(prompt).content.strip().split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", []) + [{"role": "assistant", "content": state.get("answer", "")}]
        return {"messages": msgs}

    # ── Routing functions ──────────────────────────────────────────────────────
    def route_decision(state: CapstoneState) -> str:
        r = state.get("route", "retrieve")
        if r == "tool":        return "tool"
        if r == "memory_only": return "skip"
        return "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        if (
            state.get("faithfulness", 1.0) >= FAITHFULNESS_THRESHOLD
            or state.get("eval_retries", 0) >= MAX_EVAL_RETRIES
        ):
            return "save"
        return "answer"

    # ── Graph assembly ─────────────────────────────────────────────────────────
    graph = StateGraph(CapstoneState)
    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")
    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_conditional_edges(
        "eval", eval_decision, {"answer": "answer", "save": "save"}
    )
    graph.add_edge("save", END)

    compiled = graph.compile(checkpointer=MemorySaver())
    return compiled, embedder, collection
