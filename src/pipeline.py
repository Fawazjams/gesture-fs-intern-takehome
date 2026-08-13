"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import argparse
import glob
import os
from typing import Any, Callable

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base

# A local LLM: takes a prompt, returns [{"generated_text": ...}].
LLM = Callable[[str], list[dict[str, str]]]


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


def _preview(text: str, width: int = 120) -> str:
    """Collapse a chunk onto one line, trimmed to `width` characters."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width].rstrip() + "..."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store: Any, llm: LLM, question: str) -> dict:
    """Retrieve relevant chunks and generate an answer.

    Steps:
      1. Use vector_store.similarity_search(question, k=3) to get
         the top 3 most relevant document chunks.
      2. Combine the chunk text into a single context string.
         (Hint: each chunk has a .page_content attribute)
      3. Format the PROMPT_TEMPLATE with the context and question.
      4. Pass the formatted prompt to llm(...) and extract the
         generated text from the result.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    docs = vector_store.similarity_search(question, k=3)
    sources = [doc.page_content for doc in docs]
    context = "\n\n".join(sources)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    result = llm(prompt)
    answer = result[0]["generated_text"].strip()

    return {"answer": answer, "sources": sources}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _print_result(result: dict) -> None:
    """Print retrieved sources and the generated answer."""
    print("\n📄 Sources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"  {i}. {_preview(source)}")
    print(f"\n💬 Answer: {result['answer']}\n")


def main() -> None:
    """Interactive Q&A loop.

    Steps:
      1. Build the knowledge base using build_knowledge_base()
         with the data/ directory path.
      2. Load the LLM using get_llm().
      3. Start a loop that:
         - Prompts the user for a question with input()
         - Exits if they type "quit"
         - Calls ask_question() with their input
         - Prints the retrieved sources and the answer

    Passing --query "..." answers that one question and exits instead.
    """
    parser = argparse.ArgumentParser(
        description="Ask questions about the agency's services, pricing, and process."
    )
    parser.add_argument(
        "--query",
        help="Answer a single question and exit, instead of starting the interactive loop.",
    )
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data directory not found: {os.path.abspath(data_dir)}")
    if not glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True):
        raise SystemExit(f"No .txt documents found in {os.path.abspath(data_dir)}")

    vector_store = build_knowledge_base(data_dir)

    print("Loading LLM (first run downloads ~1GB)...")
    llm = get_llm()
    print("  Done!\n")

    # Single-question mode: answer once and exit.
    if args.query:
        _print_result(ask_question(vector_store, llm, args.query))
        return

    print('Ask about our services, pricing, or process. Type "quit" to exit.\n')

    while True:
        try:
            question = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()  # leave the shell prompt on its own line
            break

        if not question.strip():
            continue
        if question.strip().lower() == "quit":
            break

        _print_result(ask_question(vector_store, llm, question))


if __name__ == "__main__":
    main()