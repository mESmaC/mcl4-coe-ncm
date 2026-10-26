"""3-Stage Corpus Curation & Pre-Filtering Cascade with `GLiNER2.5-Decide`.

Pipeline architecture (`Stage 0 -> Stage 1 -> Stage 2`):
1. **Stage 0 (Zero-Cost CPU Byte Sieve)**:
   Deterministic `<0.01 ms/doc` byte/character heuristics (length bounds, printable ratio,
   alphabetic density, max line length, and `#U$#`/`#T$#`/`#A$#` marker collision sanitization)
   that drop 40-60% of raw web/code junk (minified JS, hex/base64 dumps, HTML chrome) before
   touching the GPU.
2. **Stage 1 (`GLiNER2.5-Decide` Multi-Head Schema Gate on RTX 5070)**:
   Uses the local 340M `Gliner2ForSchemaExtraction` (`microsoft/deberta-v3-large` backbone at
   `X:\\GLiNER2.5-Decide`, `max_position_embeddings=512`) to evaluate **three classification
   heads simultaneously in a single forward pass** over a head+tail character window:
   - `domain` (single-label): expository prose, algorithmic code/math, step-by-step reasoning
     vs. SEO boilerplate, conversational fluff, or garbled fragments.
   - `reasoning_density` (single-label): high logical density vs. moderate vs. low-value filler.
   - `defects` (multi-label with threshold): truncated context, paywall/nav chrome,
     repetitive template spam, or ungrounded assertions.
3. **Stage 2 (Normalization & `.tok16` Block Packing)**:
   Only the ~15-25% of documents surviving Stages 0 and 1 undergo line unwrapping, AST/trace
   synthesis, and `#U$# -> #T$# -> #A$#` `0x8000` masked `.tok16` block packing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .data import TRACE_MARKERS, ByteTokenizer

DEFAULT_GLINER25_PATH = Path(r"X:\GLiNER2.5-Decide")

ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        "expository_prose_or_textbook",
        "algorithmic_code_or_formal_math",
        "step_by_step_reasoning_or_qa",
    }
)

ALLOWED_DENSITIES: frozenset[str] = frozenset(
    {
        "high_logical_or_explanatory_density",
        "moderate_informational_density",
    }
)


@dataclass(frozen=True)
class ByteSieveConfig:
    """Configuration for Stage-0 deterministic CPU byte/character pre-filtering."""

    min_chars: int = 120
    max_chars: int = 100_000
    min_alpha_ratio: float = 0.45
    min_printable_ratio: float = 0.96
    max_line_length: int = 1_200
    max_digit_ratio: float = 0.35
    sanitize_protocol_markers: bool = True


@dataclass(frozen=True)
class SieveDecision:
    """Result of evaluating a raw document through Stage 0 and Stage 1."""

    keep: bool
    stage_rejected: str | None
    reason: str
    domain: str | None = None
    reasoning_density: str | None = None
    defects: tuple[str, ...] = ()
    sanitized_text: str | None = None


_RAW_THINK_MARKUP_RE = re.compile(r"</?think>", re.IGNORECASE)


def sanitize_trace_markers(text: str) -> str:
    """Escape accidental occurrences of `#U$#`, `#Q$#`, `#S$#`, `#H$#`, `#T$#`, `#A$#` and strip `<think>` tags."""
    out = _RAW_THINK_MARKUP_RE.sub("", text)
    for marker in TRACE_MARKERS.values():
        if marker in out:
            # Replace `#X$#` with `#X$` so raw corpus content can never spoof a register boundary
            out = out.replace(marker, marker[:-1])
    return out



def evaluate_byte_sieve(text: str, config: ByteSieveConfig = ByteSieveConfig()) -> SieveDecision:
    """Run Stage-0 deterministic CPU byte/character checks (`<0.01 ms/doc`)."""
    if not isinstance(text, str):
        return SieveDecision(False, "stage0_byte_sieve", "non_string_input")

    stripped = text.strip()
    n = len(stripped)
    if n < config.min_chars:
        return SieveDecision(False, "stage0_byte_sieve", f"too_short ({n} < {config.min_chars})")
    if n > config.max_chars:
        return SieveDecision(False, "stage0_byte_sieve", f"too_long ({n} > {config.max_chars})")

    alpha_count = sum(ch.isalpha() for ch in stripped)
    digit_count = sum(ch.isdigit() for ch in stripped)
    printable_count = sum(ch.isprintable() or ch in "\n\r\t" for ch in stripped)

    alpha_ratio = alpha_count / n
    digit_ratio = digit_count / n
    printable_ratio = printable_count / n

    if printable_ratio < config.min_printable_ratio:
        return SieveDecision(
            False,
            "stage0_byte_sieve",
            f"low_printable_ratio ({printable_ratio:.3f} < {config.min_printable_ratio})",
        )
    if alpha_ratio < config.min_alpha_ratio:
        return SieveDecision(
            False,
            "stage0_byte_sieve",
            f"low_alpha_ratio ({alpha_ratio:.3f} < {config.min_alpha_ratio})",
        )
    if digit_ratio > config.max_digit_ratio:
        return SieveDecision(
            False,
            "stage0_byte_sieve",
            f"excessive_digit_ratio ({digit_ratio:.3f} > {config.max_digit_ratio})",
        )

    longest_line = max((len(line) for line in stripped.splitlines()), default=0)
    if longest_line > config.max_line_length:
        return SieveDecision(
            False,
            "stage0_byte_sieve",
            f"line_too_long ({longest_line} > {config.max_line_length})",
        )

    cleaned = sanitize_trace_markers(stripped) if config.sanitize_protocol_markers else stripped
    return SieveDecision(
        keep=True,
        stage_rejected=None,
        reason="passed_stage0",
        sanitized_text=cleaned,
    )


def build_head_tail_window(text: str, head_chars: int = 1_100, tail_chars: int = 500) -> str:
    """Extract a `<=512`-subword representative snippet covering both opening and closing context.

    Why head + tail:
    `GLiNER2.5-Decide` uses `microsoft/deberta-v3-large` (`max_position_embeddings = 512`).
    Inspecting the first `1,100` chars plus the final `500` chars (~350-420 subwords total)
    captures both the document's core subject matter at the top and truncation/paywall/footer
    defects at the bottom in a single 512-token encoder pass.
    """
    stripped = text.strip()
    if len(stripped) <= head_chars + tail_chars + 32:
        return stripped
    return f"{stripped[:head_chars].rstrip()}\n...\n{stripped[-tail_chars:].lstrip()}"


@dataclass
class GLiNER25GateConfig:
    """Configuration for Stage-1 `GLiNER2.5-Decide` multi-head corpus filtering."""

    model_path: Path = DEFAULT_GLINER25_PATH
    head_chars: int = 1_100
    tail_chars: int = 500
    defect_threshold: float = 0.45
    require_high_density_for_prose: bool = False
    domain_labels: tuple[str, ...] = (
        "expository_prose_or_textbook",
        "algorithmic_code_or_formal_math",
        "step_by_step_reasoning_or_qa",
        "seo_marketing_or_boilerplate",
        "conversational_chatter_or_forum_noise",
        "garbled_fragment_or_dump",
    )
    density_labels: tuple[str, ...] = (
        "high_logical_or_explanatory_density",
        "moderate_informational_density",
        "low_value_filler_or_stub",
    )
    defect_labels: tuple[str, ...] = (
        "truncated_or_incomplete",
        "paywall_or_navigation_chrome",
        "repetitive_template_spam",
        "ungrounded_assertions_without_explanation",
    )
    allowed_domains: frozenset[str] = field(default_factory=lambda: ALLOWED_DOMAINS)
    allowed_densities: frozenset[str] = field(default_factory=lambda: ALLOWED_DENSITIES)


class GLiNER25CorpusGate:
    """Stage-1 Multi-Head Corpus Decision Gate backed by `X:\\GLiNER2.5-Decide`.

    Evaluates `domain`, `reasoning_density`, and multi-label `defects` in a single
    forward pass of the 340M DeBERTa-v3-Large schema extractor. Also accepts an optional
    custom `classifier_fn` for unit testing when `gliner2` is not loaded in the current
    process.
    """

    def __init__(
        self,
        config: GLiNER25GateConfig = GLiNER25GateConfig(),
        *,
        sieve_config: ByteSieveConfig = ByteSieveConfig(),
        classifier_fn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.sieve_config = sieve_config
        self._classifier_fn = classifier_fn
        self._extractor: Any = None

    def _build_task_schema(self) -> dict[str, Any]:
        return {
            "domain": list(self.config.domain_labels),
            "reasoning_density": list(self.config.density_labels),
            "defects": {
                "labels": list(self.config.defect_labels),
                "multi_label": True,
                "cls_threshold": self.config.defect_threshold,
            },
        }

    def _load_extractor(self) -> Any:
        if self._extractor is not None:
            return self._extractor
        try:
            from gliner2 import AutoExtractor  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "The `gliner2` package is required to load `X:\\GLiNER2.5-Decide`. "
                "Install it with `pip install gliner2` or pass `classifier_fn` for testing."
            ) from exc
        self._extractor = AutoExtractor.from_pretrained(str(self.config.model_path))
        return self._extractor

    def classify_window(self, window_text: str) -> dict[str, Any]:
        schema = self._build_task_schema()
        if self._classifier_fn is not None:
            return self._classifier_fn(window_text, schema)
        extractor = self._load_extractor()
        return extractor.classify_text(window_text, schema)

    @staticmethod
    def _extract_label(raw: Any) -> str | None:
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict) and "label" in raw:
            return str(raw["label"])
        return None

    @staticmethod
    def _extract_multi_labels(raw: Any) -> tuple[str, ...]:
        if raw is None:
            return ()
        if isinstance(raw, (list, tuple)):
            out: list[str] = []
            for item in raw:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict) and "label" in item:
                    out.append(str(item["label"]))
            return tuple(out)
        return ()

    def evaluate_document(self, text: str) -> SieveDecision:
        """Run Stage 0 (CPU byte sieve) followed by Stage 1 (`GLiNER2.5-Decide` multi-head gate)."""
        stage0 = evaluate_byte_sieve(text, self.sieve_config)
        if not stage0.keep or stage0.sanitized_text is None:
            return stage0

        window = build_head_tail_window(
            stage0.sanitized_text,
            head_chars=self.config.head_chars,
            tail_chars=self.config.tail_chars,
        )
        preds = self.classify_window(window)
        domain = self._extract_label(preds.get("domain"))
        density = self._extract_label(preds.get("reasoning_density"))
        defects = self._extract_multi_labels(preds.get("defects"))

        if defects:
            return SieveDecision(
                keep=False,
                stage_rejected="stage1_gliner25_decide",
                reason=f"defects_detected ({','.join(defects)})",
                domain=domain,
                reasoning_density=density,
                defects=defects,
                sanitized_text=stage0.sanitized_text,
            )

        if domain not in self.config.allowed_domains:
            return SieveDecision(
                keep=False,
                stage_rejected="stage1_gliner25_decide",
                reason=f"disallowed_domain ({domain})",
                domain=domain,
                reasoning_density=density,
                defects=defects,
                sanitized_text=stage0.sanitized_text,
            )

        if density not in self.config.allowed_densities:
            return SieveDecision(
                keep=False,
                stage_rejected="stage1_gliner25_decide",
                reason=f"low_reasoning_density ({density})",
                domain=domain,
                reasoning_density=density,
                defects=defects,
                sanitized_text=stage0.sanitized_text,
            )

        if (
            self.config.require_high_density_for_prose
            and domain == "expository_prose_or_textbook"
            and density != "high_logical_or_explanatory_density"
        ):
            return SieveDecision(
                keep=False,
                stage_rejected="stage1_gliner25_decide",
                reason="prose_requires_high_logical_density",
                domain=domain,
                reasoning_density=density,
                defects=defects,
                sanitized_text=stage0.sanitized_text,
            )

        return SieveDecision(
            keep=True,
            stage_rejected=None,
            reason="passed_stage0_and_stage1",
            domain=domain,
            reasoning_density=density,
            defects=defects,
            sanitized_text=stage0.sanitized_text,
        )


_HARD_WRAP_RE = re.compile(r"(?<=[a-z0-9,;:])\n(?=[a-z])")


def normalize_prose_unwrapping(text: str) -> str:
    """Stage-2 normalization: unwrap hard-wrapped 80-column prose lines while preserving paragraphs/code."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return _HARD_WRAP_RE.sub(" ", normalized).strip()
