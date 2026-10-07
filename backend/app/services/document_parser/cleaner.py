import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Cleans and normalizes extracted raw text:
    - Normalizes unicode to NFKC form.
    - Strips null bytes and unusual control characters.
    - Standardizes carriage returns and newlines.
    - Collapses excessive horizontal whitespace.
    - Normalizes paragraph breaks (maximum 2 consecutive newlines).
    """
    if not text:
        return ""

    # Normalize unicode characters
    normalized = unicodedata.normalize("NFKC", text)

    # Remove null bytes and non-printable control characters except newline and tab
    sanitized = "".join(
        ch for ch in normalized
        if ch in ("\n", "\r", "\t") or (unicodedata.category(ch)[0] != "C")
    )

    # Standardize line endings to \n
    standardized_lines = sanitized.replace("\r\n", "\n").replace("\r", "\n")

    # Clean line-by-line whitespace
    cleaned_lines = []
    for line in standardized_lines.split("\n"):
        # Replace non-breaking spaces and collapse horizontal spaces
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    reconstructed = "\n".join(cleaned_lines)

    # Collapse 3 or more consecutive newlines into 2 (paragraph break)
    collapsed = re.sub(r"\n{3,}", "\n\n", reconstructed)

    return collapsed.strip()
