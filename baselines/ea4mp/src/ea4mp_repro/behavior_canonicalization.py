"""Label-agnostic canonicalization for released-style EA4MP behavior text.

The released PyCG/CFG pipeline emits local symbols qualified by the archive
extraction path.  In this reconstruction that path was stabilized with the
``sample_id`` and therefore exposed collection/label-source prefixes such as
``benign__`` and ``guo__`` to BERT.  Raw behavior artifacts remain immutable;
this module removes only that synthetic local identity immediately before
WordPiece tokenization.

External API names and behavior order are left untouched.  Local definitions
and ``<call>`` targets retain their terminal function/method name under one
constant namespace, matching the path-independent intent of the released
preprocessing code without pretending that the unreleased ``train.txt``
builder is known.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


POLICY = "ea4mp-local-identity-canonicalization-v1"
RAW_SLASH_ROOT = "/__EA4MP_ARCHIVE__/"
RAW_DOTTED_ROOT = "__EA4MP_ARCHIVE__."
CANONICAL_FILE_ROOT = "/__EA4MP_LOCAL_FILE__"
CANONICAL_SYMBOL_ROOT = "__EA4MP_LOCAL__"


class BehaviorCanonicalizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CanonicalizationAudit:
    policy: str
    file_headers: int
    local_modules: int
    local_symbol_replacements: int
    fallback_local_symbol_replacements: int
    raw_slash_root_occurrences: int
    raw_dotted_root_occurrences: int
    residual_sample_id_occurrences: int
    residual_raw_root_occurrences: int
    residual_source_tag_occurrences: int
    label_read: bool
    split_read: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_sample_id(sample_id: str) -> None:
    if (
        not sample_id
        or "/" in sample_id
        or "\\" in sample_id
        or "\n" in sample_id
        or "\r" in sample_id
        or "__" not in sample_id
    ):
        raise BehaviorCanonicalizationError("invalid sample_id for canonicalization")


def _terminal_symbol(raw_symbol: str) -> str:
    terminal = raw_symbol.rsplit(".", 1)[-1]
    if not terminal:
        raise BehaviorCanonicalizationError("empty terminal local symbol")
    return terminal


def _dotted_module_candidates(
    header: str,
    *,
    raw_slash_prefix: str,
    raw_dotted_prefix: str,
) -> tuple[str, ...]:
    relative = header[len(raw_slash_prefix) :].lstrip("/")
    if not relative:
        raise BehaviorCanonicalizationError("empty package-relative DFS file header")
    dotted = raw_dotted_prefix + "." + relative.replace("/", ".")
    if dotted.endswith(".py"):
        dotted = dotted[:-3]
    candidates = [dotted]
    # PyCG commonly drops ``.__init__`` from package module names.
    if dotted.endswith(".__init__"):
        candidates.append(dotted[: -len(".__init__")])
    return tuple(sorted(set(candidates), key=len, reverse=True))


def canonicalize_behavior_text(
    dfs_text: str,
    *,
    sample_id: str,
) -> tuple[str, CanonicalizationAudit]:
    """Remove synthetic sample identity from one package's DFS text.

    This function deliberately accepts neither a label nor a split.  Its only
    sample-specific input is used to identify the exact synthetic extraction
    root that the local runner inserted.  It fails closed if any part of that
    identity survives in the canonical text.
    """

    _validate_sample_id(sample_id)
    raw_slash_prefix = f"{RAW_SLASH_ROOT}{sample_id}"
    raw_dotted_prefix = f"{RAW_DOTTED_ROOT}{sample_id}"
    source_tag = sample_id.split("__", 1)[0] + "__"
    raw_slash_count = dfs_text.count(raw_slash_prefix)
    raw_dotted_count = dfs_text.count(raw_dotted_prefix)

    raw_lines = dfs_text.splitlines(keepends=True)
    header_to_module: dict[str, str] = {}
    module_prefixes: dict[str, str] = {}
    header_count = 0
    for raw_line in raw_lines:
        content = raw_line.rstrip("\r\n")
        if content.strip().startswith("/"):
            header = content.strip()
            if not header.startswith(raw_slash_prefix):
                raise BehaviorCanonicalizationError(
                    "DFS file header is outside the expected synthetic sample root"
                )
            header_count += 1
            if header not in header_to_module:
                module_id = f"f{len(header_to_module) + 1:04d}"
                header_to_module[header] = module_id
                for prefix in _dotted_module_candidates(
                    header,
                    raw_slash_prefix=raw_slash_prefix,
                    raw_dotted_prefix=raw_dotted_prefix,
                ):
                    previous = module_prefixes.get(prefix)
                    if previous is not None and previous != module_id:
                        raise BehaviorCanonicalizationError(
                            "two DFS headers map to the same local module prefix"
                        )
                    module_prefixes[prefix] = module_id

    # Longest-prefix replacement preserves a stable anonymous module identity
    # plus the original class/function suffix.  A conservative terminal-name
    # fallback handles local symbols emitted for imported files that have no
    # DFS header in this top-files track.
    mapped_patterns = [
        (
            re.compile(
                re.escape(prefix) + r"(?=\.|[\s()\[\]:]|$)"
            ),
            f"{CANONICAL_SYMBOL_ROOT}.{module_id}",
        )
        for prefix, module_id in sorted(
            module_prefixes.items(), key=lambda item: len(item[0]), reverse=True
        )
    ]
    fallback_pattern = re.compile(
        re.escape(raw_dotted_prefix) + r"(?:\.[^\s()\[\]:]+)*"
    )

    output_lines: list[str] = []
    symbol_replacements = 0
    fallback_replacements = 0
    for raw_line in raw_lines:
        content = raw_line.rstrip("\r\n")
        ending = raw_line[len(content) :]
        if content.strip().startswith("/"):
            module_id = header_to_module[content.strip()]
            output_lines.append(f"{CANONICAL_FILE_ROOT}/{module_id}.py{ending}")
            continue

        transformed = raw_line
        for pattern, replacement in mapped_patterns:
            transformed, count = pattern.subn(replacement, transformed)
            symbol_replacements += count

        def replace_symbol(match: re.Match[str]) -> str:
            nonlocal symbol_replacements, fallback_replacements
            symbol_replacements += 1
            fallback_replacements += 1
            raw_symbol = match.group(0)
            if raw_symbol == raw_dotted_prefix:
                return CANONICAL_SYMBOL_ROOT
            return f"{CANONICAL_SYMBOL_ROOT}.{_terminal_symbol(raw_symbol)}"

        output_lines.append(fallback_pattern.sub(replace_symbol, transformed))

    canonical = "".join(output_lines)
    residual_sample = canonical.count(sample_id)
    residual_root = canonical.count(RAW_SLASH_ROOT) + canonical.count(
        RAW_DOTTED_ROOT
    )
    residual_source_tag = canonical.count(source_tag)
    if residual_sample or residual_root or residual_source_tag:
        raise BehaviorCanonicalizationError(
            "synthetic sample identity survived behavior canonicalization: "
            f"sample_id={residual_sample}, root={residual_root}, "
            f"source_tag={residual_source_tag}"
        )
    if dfs_text.strip() and raw_slash_count == 0:
        raise BehaviorCanonicalizationError(
            "non-empty DFS text has no expected synthetic file header"
        )
    if header_count != raw_slash_count:
        raise BehaviorCanonicalizationError(
            "synthetic file-header replacement count drift"
        )

    return canonical, CanonicalizationAudit(
        policy=POLICY,
        file_headers=header_count,
        local_modules=len(header_to_module),
        local_symbol_replacements=symbol_replacements,
        fallback_local_symbol_replacements=fallback_replacements,
        raw_slash_root_occurrences=raw_slash_count,
        raw_dotted_root_occurrences=raw_dotted_count,
        residual_sample_id_occurrences=residual_sample,
        residual_raw_root_occurrences=residual_root,
        residual_source_tag_occurrences=residual_source_tag,
        label_read=False,
        split_read=False,
    )
