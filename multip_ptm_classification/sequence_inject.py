"""
Inject PTM-specific tokenizer tokens into window sequences for Mamba encoding.

Typical use: 51-residue windows use plain one-letter AAs with the modification
site at the window center. Replacing that single character with the correct
`<...>` PTM token aligns the backbone with how PTM-Mamba was trained.
"""


def inject_ptm_token_at_center(sequence: str, ptm_token: str) -> str:
    """
    Replace the middle amino acid (one letter) with `ptm_token`.

    For odd length L, index is L // 2. For even L, uses L // 2 (right-biased
    middle), matching common fixed-window slicing.
    """
    if not sequence:
        return sequence
    s = sequence.strip()
    if len(s) == 1:
        return ptm_token
    mid = len(s) // 2
    return s[:mid] + ptm_token + s[mid + 1 :]


def inject_ptm_token_at_index(sequence: str, ptm_token: str, index: int) -> str:
    """Replace the character at `index` (0-based) with `ptm_token`."""
    s = sequence
    if not s:
        return s
    if index < 0:
        index = len(s) + index
    index = max(0, min(index, len(s) - 1))
    return s[:index] + ptm_token + s[index + 1 :]
