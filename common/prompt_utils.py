"""
Prompt construction helpers shared across baselines and gap experiments.
Keep the template IDENTICAL across all experiments — only the retrieval /
caching strategy should change between runs, so results are comparable.
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to retrieved documents and "
    "memories from earlier in the conversation. Use them to answer accurately. "
    "If the answer is not in the provided context, say you don't know."
)


def format_context_block(label: str, items: list[str]) -> str:
    if not items:
        return ""
    joined = "\n".join(f"- {item}" for item in items)
    return f"\n\n[{label}]\n{joined}"


def build_prompt(history: list[str], retrieved: list[str], query: str,
                  memories: list[str] | None = None) -> str:
    """
    history   : prior turns (plain text, already summarized/truncated upstream)
    retrieved : RAG-retrieved document chunks for this turn
    memories  : agent-memory items retrieved for this turn (optional)
    query     : the current user query
    """
    parts = [SYSTEM_PROMPT]
    if history:
        parts.append(format_context_block("Conversation so far", history))
    if memories:
        parts.append(format_context_block("Relevant memories", memories))
    if retrieved:
        parts.append(format_context_block("Retrieved documents", retrieved))
    parts.append(f"\n\n[Current question]\n{query}\n\nAnswer concisely in 1-2 sentences, then write <END>.\n\nAnswer:")
    return "".join(parts)
