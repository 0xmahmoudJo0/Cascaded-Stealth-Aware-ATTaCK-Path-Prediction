"""
Technique embeddings training and management.

This module trains Word2Vec embeddings on technique sequences and builds
ANN indices for fast similarity search during prediction.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
# FAISS removed - using NPY embeddings only
from tqdm import tqdm

from . import config as cfg
from .io import load_sequences_for_source, save_numpy, save_csv, ensure_dir


def train_word2vec(sequences: List[List[str]], model_path: Path) -> Word2Vec:
    """
    Train Word2Vec model on technique sequences.
    
    Args:
        sequences: List of technique sequences
        model_path: Path to save the trained model
        
    Returns:
        Trained Word2Vec model
    """
    print("Training Word2Vec model...")
    print(f"Parameters: {cfg.WORD2VEC_PARAMS}")
    
    # Ensure sequences are in the correct format (list of lists of strings)
    training_sequences = [list(map(str, seq)) for seq in sequences]
    
    # Train Word2Vec model
    model = Word2Vec(sentences=training_sequences, **cfg.WORD2VEC_PARAMS)
    
    # Save model
    ensure_dir(model_path.parent)
    model.save(str(model_path))
    print(f"Saved Word2Vec model to {model_path}")
    
    return model


def extract_embeddings(model: Word2Vec) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Extract embeddings and create technique index from Word2Vec model.
    
    Args:
        model: Trained Word2Vec model
        
    Returns:
        Tuple of (embeddings_array, tech_to_idx_mapping)
    """
    print("Extracting embeddings...")
    
    # Get vocabulary
    vocab = list(model.wv.key_to_index.keys())
    vocab_size = len(vocab)
    vector_size = model.wv.vector_size
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vector size: {vector_size}")
    
    # Create technique to index mapping
    tech_to_idx = {tech: idx for idx, tech in enumerate(vocab)}
    
    # Extract embeddings
    embeddings = np.zeros((vocab_size, vector_size), dtype=np.float32)
    
    for tech, idx in tqdm(tech_to_idx.items(), desc="Extracting embeddings"):
        embeddings[idx] = model.wv[tech]
    
    return embeddings, tech_to_idx


def _l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Safely apply L2 normalization row-wise."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def train_content_embeddings(tech_to_idx: Dict[str, int], stix_json_path: Path) -> np.ndarray:
    """Generate content embeddings for techniques using MITRE STIX data."""

    print("Training content embeddings from MITRE STIX JSON...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = model.get_sentence_embedding_dimension()

    content_embeddings = np.zeros((len(tech_to_idx), embedding_dim), dtype=np.float32)
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    if not stix_json_path.exists():
        print(f"STIX file not found at {stix_json_path}. Returning zero content embeddings.")
        return content_embeddings

    try:
        with stix_json_path.open('r', encoding='utf-8') as handle:
            stix_data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to load STIX content from {stix_json_path}: {exc}. Using zero embeddings.")
        return content_embeddings

    if isinstance(stix_data, dict):
        objects = stix_data.get('objects', []) or []
    elif isinstance(stix_data, list):
        objects = stix_data
    else:
        objects = []

    attack_patterns = [
        obj for obj in objects
        if isinstance(obj, dict) and obj.get('type') == 'attack-pattern'
    ]

    attack_texts: Dict[str, str] = {}
    for obj in attack_patterns:
        technique_id = None
        for reference in obj.get('external_references', []):
            if not isinstance(reference, dict):
                continue
            source_name = reference.get('source_name', '') or ''
            if 'mitre-attack' in source_name and reference.get('external_id'):
                technique_id = reference['external_id']
                break
        if not technique_id:
            continue
        name = obj.get('name', '') or ''
        description = obj.get('description', '') or ''
        attack_texts[technique_id] = f"{name}: {description}".strip()

    texts_to_encode: List[str] = []
    indices: List[int] = []
    for tech, idx in sorted(tech_to_idx.items(), key=lambda item: item[1]):
        text = attack_texts.get(tech)
        if text:
            texts_to_encode.append(text)
            indices.append(idx)

    if texts_to_encode:
        encoded = model.encode(
            texts_to_encode,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        for encoded_idx, emb_idx in enumerate(indices):
            content_embeddings[emb_idx] = encoded[encoded_idx]

    missing_indices = [idx for idx in range(len(tech_to_idx)) if idx not in indices]
    if missing_indices:
        noise = rng.normal(scale=1e-3, size=(len(missing_indices), embedding_dim)).astype(np.float32)
        for noise_idx, emb_idx in enumerate(missing_indices):
            content_embeddings[emb_idx] = noise[noise_idx]

    return content_embeddings


def build_faiss_index(embeddings: np.ndarray) -> None:
    """DEPRECATED - FAISS removed. Using NPY embeddings only."""
    pass


def save_embeddings_and_index(embeddings: np.ndarray, tech_to_idx: Dict[str, int], 
                             model: Word2Vec, embeddings_path: Path, 
                             index_path: Path, tech_index_path: Path) -> None:
    """
    Save embeddings, index, and technique mapping.
    
    Args:
        embeddings: Embeddings array
        tech_to_idx: Technique to index mapping
        model: Word2Vec model
        embeddings_path: Path to save embeddings
        index_path: Path to save index
        tech_index_path: Path to save technique index
    """
    print("Saving embeddings and index...")
    
    # Save embeddings
    save_numpy(embeddings, embeddings_path)
    print(f"Saved embeddings to {embeddings_path}")
    
    # Save technique index
    save_csv(pd.DataFrame([
        {'technique_id': tech, 'index': idx}
        for tech, idx in tech_to_idx.items()
    ]), tech_index_path)
    print(f"Saved technique index to {tech_index_path}")
    
    # FAISS index removed - using NPY embeddings only
    print("Note: FAISS index not generated - using NPY format only")


def load_embeddings_and_index(embeddings_path: Path, index_path: Path, 
                             tech_index_path: Path) -> Tuple[np.ndarray, Dict[str, int], any]:
    """
    Load embeddings, index, and technique mapping.
    
    Args:
        embeddings_path: Path to embeddings file
        index_path: Path to index file
        tech_index_path: Path to technique index file
        
    Returns:
        Tuple of (embeddings, tech_to_idx, index)
    """
    print("Loading embeddings and index...")
    
    # Load embeddings
    try:
        embeddings = np.load(embeddings_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Embeddings file not found at {embeddings_path}. "
            "Run `python -m src.chain_paths.cli emb` to generate embeddings before building profiles."
        ) from exc
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Load technique index
    try:
        tech_df = pd.read_csv(tech_index_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Technique index not found at {tech_index_path}. "
            "Run `python -m src.chain_paths.cli emb` to generate embeddings before building profiles."
        ) from exc
    tech_to_idx = dict(zip(tech_df['technique_id'], tech_df['index']))
    print(f"Loaded technique index: {len(tech_to_idx)} techniques")
    
    # FAISS index removed - returning None
    return embeddings, tech_to_idx, None


def get_embedding_similarity(embeddings: np.ndarray, tech_to_idx: Dict[str, int], 
                           index: any, context_techs: List[str], 
                           candidate_tech: str) -> float:
    """
    Get embedding similarity between context and candidate.
    
    Args:
        embeddings: Embeddings array
        tech_to_idx: Technique to index mapping
        index: ANN index
        context_techs: Context techniques
        candidate_tech: Candidate technique
        
    Returns:
        Cosine similarity score
    """
    # Get context embedding (mean of context techniques)
    context_indices = [tech_to_idx[tech] for tech in context_techs if tech in tech_to_idx]
    
    if not context_indices:
        return 0.0
    
    context_embedding = np.mean(embeddings[context_indices], axis=0)
    
    # Get candidate embedding
    if candidate_tech not in tech_to_idx:
        return 0.0
    
    candidate_idx = tech_to_idx[candidate_tech]
    candidate_embedding = embeddings[candidate_idx]
    
    # Compute cosine similarity
    similarity = np.dot(context_embedding, candidate_embedding) / (
        np.linalg.norm(context_embedding) * np.linalg.norm(candidate_embedding)
    )
    
    return float(similarity)


def get_top_similar_techniques(embeddings: np.ndarray, tech_to_idx: Dict[str, int], 
                              index: any, context_techs: List[str], 
                              top_k: int = 50) -> List[str]:
    """
    Get top-K most similar techniques to context using embedding similarity.
    
    Args:
        embeddings: Embeddings array
        tech_to_idx: Technique to index mapping
        index: Unused (kept for API compatibility, FAISS removed)
        context_techs: Context techniques
        top_k: Number of top techniques to return
        
    Returns:
        List of top similar techniques
    """
    # Get context embedding (mean of context techniques)
    context_indices = [tech_to_idx[tech] for tech in context_techs if tech in tech_to_idx]
    
    if not context_indices:
        return []
    
    context_embedding = np.mean(embeddings[context_indices], axis=0)
    
    # Compute cosine similarity with all techniques
    # L2 normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-12)
    context_norm = context_embedding / (np.linalg.norm(context_embedding) + 1e-12)
    
    # Compute similarities
    similarities = np.dot(normalized, context_norm)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Convert indices to techniques
    idx_to_tech = {idx: tech for tech, idx in tech_to_idx.items()}
    similar_techs = [idx_to_tech[idx] for idx in top_indices if idx in idx_to_tech]
    
    return similar_techs


def train_embeddings(sequences: List[List[str]]) -> None:
    """
    Train embeddings and save all artifacts.
    
    Args:
        sequences: List of technique sequences
    """
    print("Training technique embeddings...")
    
    # Train Word2Vec model
    model = train_word2vec(sequences, cfg.WORD2VEC_MODEL)
    
    # Extract embeddings
    embeddings, tech_to_idx = extract_embeddings(model)

    # Train content embeddings
    content_embeddings = train_content_embeddings(tech_to_idx, cfg.ENTERPRISE_ATTACK_JSON)

    # Normalize and fuse Word2Vec with content embeddings
    word2vec_normalized = _l2_normalize_rows(embeddings)
    content_normalized = _l2_normalize_rows(content_embeddings)
    fused_embeddings = np.concatenate([word2vec_normalized, content_normalized], axis=1)

    # Save embeddings and index
    index_path = cfg.TECH_EMBEDDINGS_NPY.parent / "emb_index.faiss"
    save_embeddings_and_index(
        fused_embeddings, tech_to_idx, model,
        cfg.TECH_EMBEDDINGS_NPY, index_path, cfg.TECH_INDEX_CSV
    )
    
    print("Embeddings training complete!")


def main():
    """Main function to train embeddings."""
    # Load sequences
    sequences = load_sequences_for_source(cfg.DEFAULT_DATA_SOURCE)
    print(f"Loaded {len(sequences)} sequences")
    
    # Train embeddings
    train_embeddings(sequences)


if __name__ == "__main__":
    main()
