"""Citation IDs parsed from answer text."""
import re
CITATION_TOKEN = re.compile(r"\A[A-Za-z][A-Za-z0-9_-]*\d\Z")
_BRACKETS = re.compile(r"\[([^\[\]]*)\]")
_INDEX_SUBTRACTION = re.compile(r"\A[A-Za-z]-[0-9]+\Z")
_VARIABLE_PREFIX = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*\Z")

def extract_citations(answer: str) -> list[str]:
    """Read citations from the answer text itself; never trust a self-report."""
    found: set[str] = set()
    text = str(answer)
    for match in _BRACKETS.finditer(text):
        token = match.group(1).strip()
        if text.startswith("(", match.end()):
            continue
        if token and len(token) <= 160 and CITATION_TOKEN.match(token):
            # AES array notation such as w[i-4] is not a source citation.
            # Bare [i-4] and variable[unknown_id1] remain citation candidates.
            if _INDEX_SUBTRACTION.fullmatch(token) and _VARIABLE_PREFIX.search(text[:match.start()]):
                continue
            found.add(token)
    return sorted(found)
