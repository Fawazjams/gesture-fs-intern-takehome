"""Additional tests for the Q&A Pipeline.

Kept separate from test_pipeline.py so the provided suite stays untouched.

Most of these swap in a stub LLM rather than flan-t5: the prompt-assembly
logic is what we own, and testing it without a 1GB model keeps the suite fast.

Run: pytest tests/ -v
"""

import os

import pytest

from src.knowledge_base import build_knowledge_base
from src.pipeline import PROMPT_TEMPLATE, _preview, ask_question

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# flan-t5-base truncates input at 512 tokens, and truncation drops the TAIL
# of the prompt -- which is exactly where the question sits. If a prompt ever
# exceeds this, the model silently stops seeing the question.
MAX_INPUT_TOKENS = 512


class StubLLM:
    """Stands in for get_llm(): records its prompt, returns a canned answer."""

    def __init__(self, generated_text: str = "stub answer"):
        self.generated_text = generated_text
        self.last_prompt: str | None = None

    def __call__(self, prompt: str) -> list[dict[str, str]]:
        self.last_prompt = prompt
        return [{"generated_text": self.generated_text}]


@pytest.fixture(scope="module")
def vector_store():
    """Build the vector store once for all tests in this module."""
    return build_knowledge_base(DATA_DIR)


# ────────────────────────────────
# _preview formatting
# ────────────────────────────────
class TestPreview:
    def test_collapses_newlines_and_runs_of_whitespace(self):
        assert _preview("GROWTH\n\nPACKAGE   —  $5,500") == "GROWTH PACKAGE — $5,500"

    def test_short_text_is_unchanged(self):
        assert _preview("short chunk") == "short chunk"

    def test_long_text_is_truncated_with_ellipsis(self):
        result = _preview("x" * 500, width=20)
        assert result == "x" * 20 + "..."

    def test_truncates_at_the_requested_width(self):
        assert len(_preview("word " * 100, width=40)) <= 43  # 40 + "..."


# ────────────────────────────────
# Prompt assembly
# ────────────────────────────────
class TestPromptAssembly:
    def test_question_reaches_the_prompt(self, vector_store):
        """The question must survive into the prompt, not just the context."""
        llm = StubLLM()
        ask_question(vector_store, llm, "How much is the Starter package?")
        assert "How much is the Starter package?" in llm.last_prompt

    def test_retrieved_sources_reach_the_prompt(self, vector_store):
        llm = StubLLM()
        result = ask_question(vector_store, llm, "Do you offer SEO services?")
        for source in result["sources"]:
            assert source in llm.last_prompt

    def test_no_unfilled_template_placeholders(self, vector_store):
        llm = StubLLM()
        ask_question(vector_store, llm, "How does onboarding work?")
        assert "{context}" not in llm.last_prompt
        assert "{question}" not in llm.last_prompt

    def test_answer_is_stripped(self, vector_store):
        llm = StubLLM(generated_text="  $2,500/month \n")
        result = ask_question(vector_store, llm, "How much is the Starter package?")
        assert result["answer"] == "$2,500/month"


# ────────────────────────────────
# Retrieval
# ────────────────────────────────
class TestRetrievalExtra:
    def test_retrieves_exactly_three_chunks(self, vector_store):
        result = ask_question(vector_store, StubLLM(), "What services do you offer?")
        assert len(result["sources"]) == 3

    def test_sources_are_non_empty_strings(self, vector_store):
        result = ask_question(vector_store, StubLLM(), "What tools do you use?")
        assert all(isinstance(s, str) and s.strip() for s in result["sources"])

    def test_retrieves_onboarding_content(self, vector_store):
        result = ask_question(vector_store, StubLLM(), "How does onboarding work?")
        sources_text = " ".join(result["sources"]).lower()
        assert "onboarding" in sources_text or "kickoff" in sources_text

    def test_repeated_question_is_deterministic(self, vector_store):
        q = "Can I cancel my contract?"
        assert (
            ask_question(vector_store, StubLLM(), q)["sources"]
            == ask_question(vector_store, StubLLM(), q)["sources"]
        )


# ────────────────────────────────
# Prompt length guard
# ────────────────────────────────
class TestPromptFitsModelLimit:
    """Regression guard: three 500-char chunks plus the template must stay
    under flan-t5's 512-token input cap, or the question gets truncated away."""

    @pytest.mark.parametrize(
        "question",
        [
            "How much is the Starter package?",
            "What are your PPC management fees?",
            "Do you offer SEO services?",
            "Can I cancel my contract?",
            "What services do you offer?",
        ],
    )
    def test_prompt_stays_under_token_limit(self, vector_store, question):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

        llm = StubLLM()
        ask_question(vector_store, llm, question)
        n_tokens = len(tokenizer(llm.last_prompt)["input_ids"])

        assert n_tokens <= MAX_INPUT_TOKENS, (
            f"Prompt is {n_tokens} tokens, over the {MAX_INPUT_TOKENS} cap -- "
            "the question would be truncated away before the model sees it"
        )
