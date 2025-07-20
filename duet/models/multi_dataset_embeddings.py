"""Multi-dataset column embedding strategies for transferable time series models."""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..utils import get_project_root


class FrozenBERTColumnEmbedder(nn.Module):
    """Frozen BERT embeddings with efficient lookup for multi-dataset training.

    Key advantages:
    - Pre-compute embeddings once, reuse across datasets
    - Minimal model size impact (just lookup table + projection)
    - Auto-expand for new columns without retraining
    - Semantic understanding for better transferability
    """

    def __init__(
        self,
        target_dim: int = 1,
        bert_model: str = "bert-base-uncased",
        cache_dir: Optional[Path] = None,
        tokenization_strategy: str = "split_underscore_camel",
        aggregation_strategy: str = "mean",
        device: Optional[str] = None,
    ):
        """Initialize frozen BERT column embedder.

        Args:
            target_dim: Target dimension for embeddings (usually 1)
            bert_model: BERT model to use for encoding
            cache_dir: Directory to cache pre-computed embeddings
            tokenization_strategy: How to process column names
            aggregation_strategy: How to aggregate token embeddings
            device: Device for computations
        """
        super().__init__()

        self.target_dim = target_dim
        self.bert_model = bert_model
        self.tokenization_strategy = tokenization_strategy
        self.aggregation_strategy = aggregation_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Setup cache directory
        if cache_dir is None:
            cache_dir = get_project_root() / "cache" / "column_embeddings"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache files
        self.embeddings_cache_file = (
            self.cache_dir / f"bert_embeddings_{bert_model.replace('/', '_')}.pkl"
        )
        self.metadata_cache_file = (
            self.cache_dir / f"metadata_{bert_model.replace('/', '_')}.json"
        )

        # Initialize BERT (will be used only for computing new embeddings)
        self._bert = None
        self._tokenizer = None

        # Load cached embeddings
        self.embedding_cache = {}
        self.metadata = {"bert_dim": 768}  # Default for BERT-base
        self._load_cache()

        # Get BERT dimension and setup projection
        bert_dim = self.metadata["bert_dim"]
        self.projection = nn.Linear(bert_dim, target_dim)

        # Registry for current dataset columns
        self.current_columns: List[str] = []
        self.current_embeddings: Optional[torch.Tensor] = None

    def _load_cache(self):
        """Load cached embeddings and metadata."""
        if self.embeddings_cache_file.exists():
            try:
                with open(self.embeddings_cache_file, "rb") as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"Warning: Could not load embedding cache: {e}")
                self.embedding_cache = {}

        if self.metadata_cache_file.exists():
            try:
                with open(self.metadata_cache_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata cache: {e}")
                self.metadata = {"bert_dim": 768}

    def _save_cache(self):
        """Save cached embeddings and metadata."""
        try:
            with open(self.embeddings_cache_file, "wb") as f:
                pickle.dump(self.embedding_cache, f)

            with open(self.metadata_cache_file, "w") as f:
                json.dump(self.metadata, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _init_bert(self):
        """Initialize BERT model and tokenizer (lazy loading)."""
        if self._bert is None:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self._tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
            self._bert = AutoModel.from_pretrained(self.bert_model)
            self._bert = self._bert.to(self.device)
            self._bert.eval()

            # Update metadata with actual BERT dimension
            self.metadata["bert_dim"] = self._bert.config.hidden_size

    def _process_column_name(self, column_name: str) -> str:
        """Process column name according to tokenization strategy."""
        import re

        if self.tokenization_strategy == "as_is":
            return column_name.lower()

        elif self.tokenization_strategy == "split_underscore":
            tokens = column_name.split("_")
            return " ".join(token.lower() for token in tokens if token)

        elif self.tokenization_strategy == "split_underscore_camel":
            # Split on underscores first
            parts = column_name.split("_")

            # Then split each part on camelCase
            all_tokens = []
            for part in parts:
                if not part:
                    continue

                # Split camelCase using regex
                camel_tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)", part)
                if not camel_tokens:
                    camel_tokens = [part]

                all_tokens.extend(camel_tokens)

            return " ".join(token.lower() for token in all_tokens if token)

        else:
            raise ValueError(
                f"Unknown tokenization strategy: {self.tokenization_strategy}"
            )

    def _compute_bert_embedding(self, column_name: str) -> torch.Tensor:
        """Compute BERT embedding for a single column name."""
        self._init_bert()

        # Check if already cached
        cache_key = (
            f"{column_name}_{self.tokenization_strategy}_{self.aggregation_strategy}"
        )
        if cache_key in self.embedding_cache:
            return torch.tensor(self.embedding_cache[cache_key], device=self.device)

        # Process column name
        processed_text = self._process_column_name(column_name)

        # Tokenize
        with torch.no_grad():
            tokenized = self._tokenizer(
                processed_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            # Get BERT embedding
            outputs = self._bert(**tokenized)
            hidden_states = outputs.last_hidden_state  # [1, seq_len, bert_dim]

            # Aggregate tokens
            if self.aggregation_strategy == "mean":
                attention_mask = tokenized["attention_mask"]
                masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
                summed = masked_embeddings.sum(dim=1)
                lengths = attention_mask.sum(dim=1, keepdim=True)
                pooled = summed / lengths  # [1, bert_dim]
            elif self.aggregation_strategy == "cls":
                pooled = hidden_states[:, 0, :]  # [1, bert_dim]
            elif self.aggregation_strategy == "max":
                pooled = hidden_states.max(dim=1)[0]  # [1, bert_dim]
            else:
                raise ValueError(
                    f"Unknown aggregation strategy: {self.aggregation_strategy}"
                )

            # Cache the result (move to CPU for storage)
            embedding = pooled.squeeze(0).cpu()  # [bert_dim]
            self.embedding_cache[cache_key] = embedding.numpy()

            return pooled.squeeze(0)  # [bert_dim]

    def set_columns(self, column_names: List[str]) -> None:
        """Set column names for current dataset and precompute embeddings.

        Args:
            column_names: List of column names for the current dataset
        """
        self.current_columns = column_names

        # Compute embeddings for all columns
        embeddings = []
        new_embeddings_count = 0

        # Get target device from projection layer
        projection_device = next(self.projection.parameters()).device

        for col in column_names:
            cache_key = (
                f"{col}_{self.tokenization_strategy}_{self.aggregation_strategy}"
            )

            if cache_key in self.embedding_cache:
                # Use cached embedding - move to projection device
                emb = torch.tensor(
                    self.embedding_cache[cache_key], device=projection_device
                )
            else:
                # Compute new embedding
                emb = self._compute_bert_embedding(col)
                # Move to projection device
                emb = emb.to(projection_device)
                new_embeddings_count += 1

            embeddings.append(emb)

        # Stack into tensor [num_columns, bert_dim] - already on correct device
        bert_embeddings = torch.stack(embeddings)

        # Project to target dimension
        with torch.no_grad():
            self.current_embeddings = self.projection(
                bert_embeddings
            )  # [num_columns, target_dim]

        # Save cache if we computed new embeddings
        if new_embeddings_count > 0:
            print(f"Computed {new_embeddings_count} new column embeddings")
            self._save_cache()

        print(f"Ready with embeddings for {len(column_names)} columns")

    def get_embeddings(self) -> torch.Tensor:
        """Get embeddings for current columns.

        Returns:
            Column embeddings [num_columns, target_dim]
        """
        if self.current_embeddings is None:
            raise RuntimeError("No columns set. Call set_columns() first.")

        return self.current_embeddings

    def forward(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Create column embeddings for a batch.

        Args:
            batch_size: Batch size
            sequence_length: Sequence length

        Returns:
            Column embeddings [batch_size, num_columns, sequence_length]
        """
        if self.current_embeddings is None:
            raise RuntimeError("No columns set. Call set_columns() first.")

        # Get embeddings [num_columns, target_dim] and ensure on correct device
        projection_device = next(self.projection.parameters()).device
        column_embs = self.current_embeddings.to(projection_device)

        # Expand to batch and time dimensions
        column_embs = column_embs.unsqueeze(0).unsqueeze(
            2
        )  # [1, num_columns, 1, target_dim]
        column_embs = column_embs.repeat(
            batch_size, 1, sequence_length, 1
        )  # [B, C, T, target_dim]

        # Transpose and squeeze if needed
        column_embs = column_embs.permute(0, 1, 3, 2)  # [B, C, target_dim, T]

        if self.target_dim == 1:
            column_embs = column_embs.squeeze(2)  # [B, C, T]

        return column_embs

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        return {
            "cached_embeddings": len(self.embedding_cache),
            "cache_size_mb": self.embeddings_cache_file.stat().st_size / 1024 / 1024
            if self.embeddings_cache_file.exists()
            else 0,
            "bert_model": self.bert_model,
            "bert_dim": self.metadata["bert_dim"],
        }


class AutoExpandingEmbedder(nn.Module):
    """Auto-expanding embedding layer that grows vocabulary as new columns are encountered."""

    def __init__(
        self,
        initial_vocab_size: int = 1000,
        embedding_dim: int = 32,
        target_dim: int = 1,
        growth_factor: float = 1.5,
    ):
        """Initialize auto-expanding embedder.

        Args:
            initial_vocab_size: Initial vocabulary size
            embedding_dim: Embedding dimension
            target_dim: Target output dimension
            growth_factor: Factor by which to grow vocabulary when needed
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.target_dim = target_dim
        self.growth_factor = growth_factor

        # Initialize embedding table
        self.embedding = nn.Embedding(initial_vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, target_dim)

        # Column name to index mapping
        self.column_to_idx = {}
        self.idx_to_column = {}
        self.next_idx = 0

        # Current dataset columns
        self.current_columns = []
        self.current_indices = []

    def _expand_vocabulary(self, new_size: int):
        """Expand the embedding vocabulary to new size."""
        old_size = self.embedding.num_embeddings

        if new_size <= old_size:
            return

        print(f"Expanding vocabulary from {old_size} to {new_size}")

        # Create new larger embedding
        new_embedding = nn.Embedding(new_size, self.embedding_dim)

        # Copy old weights
        with torch.no_grad():
            new_embedding.weight[:old_size] = self.embedding.weight
            # Initialize new embeddings with small random values
            nn.init.normal_(new_embedding.weight[old_size:], mean=0, std=0.01)

        # Replace old embedding
        self.embedding = new_embedding

    def set_columns(self, column_names: List[str]) -> None:
        """Set columns for current dataset, expanding vocabulary if needed."""
        self.current_columns = column_names
        self.current_indices = []

        # Check if we need to expand vocabulary
        new_columns = [col for col in column_names if col not in self.column_to_idx]
        required_size = self.next_idx + len(new_columns)

        if required_size > self.embedding.num_embeddings:
            new_vocab_size = max(
                int(self.embedding.num_embeddings * self.growth_factor), required_size
            )
            self._expand_vocabulary(new_vocab_size)

        # Assign indices to new columns
        for col in new_columns:
            self.column_to_idx[col] = self.next_idx
            self.idx_to_column[self.next_idx] = col
            self.next_idx += 1

        # Get indices for current columns
        self.current_indices = [self.column_to_idx[col] for col in column_names]
        print(
            f"Set {len(column_names)} columns (vocabulary size: {self.embedding.num_embeddings})"
        )

    def get_embeddings(self) -> torch.Tensor:
        """Get embeddings for current columns."""
        indices = torch.tensor(
            self.current_indices, dtype=torch.long, device=self.embedding.weight.device
        )
        embs = self.embedding(indices)  # [num_columns, embedding_dim]
        return self.projection(embs)  # [num_columns, target_dim]

    def forward(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Create column embeddings for a batch."""
        # Get embeddings [num_columns, target_dim]
        column_embs = self.get_embeddings()

        # Expand to batch and time dimensions
        column_embs = column_embs.unsqueeze(0).unsqueeze(2)
        column_embs = column_embs.repeat(batch_size, 1, sequence_length, 1)
        column_embs = column_embs.permute(0, 1, 3, 2)

        if self.target_dim == 1:
            column_embs = column_embs.squeeze(2)

        return column_embs


def create_multi_dataset_embedder(
    strategy: str = "frozen_bert", target_dim: int = 1, **kwargs
) -> Union[FrozenBERTColumnEmbedder, AutoExpandingEmbedder]:
    """Factory function for multi-dataset column embedders.

    Args:
        strategy: 'frozen_bert' or 'auto_expanding'
        target_dim: Target embedding dimension
        **kwargs: Additional arguments for specific embedder

    Returns:
        Configured embedder instance
    """
    if strategy == "frozen_bert":
        return FrozenBERTColumnEmbedder(target_dim=target_dim, **kwargs)
    elif strategy == "auto_expanding":
        return AutoExpandingEmbedder(target_dim=target_dim, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    # Test frozen BERT embedder
    print("Testing Frozen BERT Embedder...")
    embedder = create_multi_dataset_embedder("frozen_bert")

    # Test with sample column names
    test_columns = ["user_id", "temperature", "getUserAccountBalance", "OT", "sensor_1"]
    embedder.set_columns(test_columns)

    # Test forward pass
    batch_size, seq_len = 32, 96
    embeddings = embedder(batch_size, seq_len)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Cache stats: {embedder.get_cache_stats()}")

    # Test auto-expanding embedder
    print("\nTesting Auto-Expanding Embedder...")
    auto_embedder = create_multi_dataset_embedder("auto_expanding")
    auto_embedder.set_columns(test_columns)

    embeddings2 = auto_embedder(batch_size, seq_len)
    print(f"Embeddings shape: {embeddings2.shape}")
