"""
Map dataset folder names (e.g. acet_k) to exact PTM-Mamba tokenizer spellings.

Strings must match entries understood by PTMTokenizer / trie (see
protein_lm/tokenizer/tokenizer.py). Extend DEFAULT_PTM_TOKEN_MAP or pass
--ptm_token explicitly in extract_features_multip.py for folders not listed.
"""

from typing import Dict, Optional

# Folder name -> full PTM token string as used in training PTM-Mamba.
DEFAULT_PTM_TOKEN_MAP: Dict[str, str] = {
    # Lysine acetylation (example from README / weekly notes)
    "acet_k": "<N6-acetyllysine>",
    "acetyl_k": "<N6-acetyllysine>",
    # Phosphorylation by residue (adjust folder names to match your CSV layout)
    "phos_s": "<Phosphoserine>",
    "phos_t": "<Phosphothreonine>",
    "phos_y": "<Phosphotyrosine>",
    "phospho_s": "<Phosphoserine>",
    "phospho_t": "<Phosphothreonine>",
    "phospho_y": "<Phosphotyrosine>",
    # N-linked glycan on Asn (example token from vocab)
    "nglyc": "<N-linked (GlcNAc...) asparagine>",
    "n_glyc": "<N-linked (GlcNAc...) asparagine>",
    # Ubiquitylation-related lysine modifications if you use separate folders
    "succ_k": "<N6-succinyllysine>",
    "carboxy_k": "<N6-carboxylysine>",
}


def get_ptm_token_for_folder(ptm_folder: str, override: Optional[str] = None) -> str:
    if override:
        return override
    key = ptm_folder.strip()
    if key not in DEFAULT_PTM_TOKEN_MAP:
        raise KeyError(
            f"No default PTM token for folder '{ptm_folder}'. "
            f"Add it to DEFAULT_PTM_TOKEN_MAP in multip_ptm_classification/ptm_token_map.py "
            f"or pass --ptm_token explicitly."
        )
    return DEFAULT_PTM_TOKEN_MAP[key]
