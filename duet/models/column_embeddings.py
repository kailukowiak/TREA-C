"""Column embedding module for incorporating column semantic information."""

import os
import re
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ColumnEmbedding(nn.Module):
    """Generate embeddings for column names using simple learned embeddings.

    This module creates lightweight embeddings for column names that can be
    incorporated into the transformer architecture alongside value and mask channels.
    """

    def __init__(
        self,
        column_names: list[str],
        target_dim: int,
        embedding_dim: int = 32,
        use_bert: bool = False,
        bert_model: str = "bert-base-uncased",
        tokenization_strategy: str = "split_underscore_camel",
        aggregation_strategy: str = "mean",
        freeze_bert: bool = True,
        device: str | None = None,
    ):
        """Initialize column embedding module.

        Args:
            column_names: List of column names to encode
            target_dim: Target embedding dimension to match value/mask channels
            embedding_dim: Dimension for simple learned embeddings (ignored if use_bert=True)
            use_bert: Whether to use BERT embeddings or simple learned embeddings
            bert_model: BERT model name/path from HuggingFace (only if use_bert=True)
            tokenization_strategy: How to tokenize column names (only if use_bert=True)
            aggregation_strategy: How to aggregate multi-token embeddings (only if use_bert=True)
            freeze_bert: Whether to freeze BERT parameters (only if use_bert=True)
            device: Device to load models on
        """
        super().__init__()

        self.column_names = column_names
        self.target_dim = target_dim
        self.use_bert = use_bert
        self.device = device or "cpu"  # Default to CPU for compatibility

        if use_bert:
            # BERT-based approach
            self.tokenization_strategy = tokenization_strategy
            self.aggregation_strategy = aggregation_strategy

            # Load BERT model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.bert = AutoModel.from_pretrained(bert_model)

            # Move BERT to specified device
            self.bert = self.bert.to(self.device)

            # Freeze BERT parameters if requested
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False

            # Get BERT embedding dimension
            bert_dim = self.bert.config.hidden_size

            # Project BERT embeddings to target dimension
            self.projection = nn.Linear(bert_dim, target_dim)

            # Pre-compute column embeddings
            self._precompute_embeddings()
        else:
            # Simple learned embeddings approach
            num_columns = len(column_names)

            # Create simple embedding layer
            self.simple_embedding = nn.Embedding(num_columns, embedding_dim)

            # Project to target dimension
            self.projection = nn.Linear(embedding_dim, target_dim)

            # Create column indices
            self.register_buffer(
                "column_indices", torch.arange(num_columns, dtype=torch.long)
            )

    def _precompute_embeddings(self):
        """Pre-compute embeddings for all column names."""
        self.eval()

        with torch.no_grad():
            # Process column names
            processed_texts = [
                self._process_column_name(name) for name in self.column_names
            ]

            # Tokenize
            tokenized = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            # Move to device
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            # Get BERT embeddings
            outputs = self.bert(**tokenized)
            hidden_states = outputs.last_hidden_state  # [num_cols, seq_len, bert_dim]

            # Aggregate tokens
            if self.aggregation_strategy == "mean":
                # Mean pooling (excluding padding tokens)
                attention_mask = tokenized["attention_mask"]
                masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
                summed = masked_embeddings.sum(dim=1)
                lengths = attention_mask.sum(dim=1, keepdim=True)
                pooled = summed / lengths  # [num_cols, bert_dim]
            elif self.aggregation_strategy == "cls":
                pooled = hidden_states[:, 0, :]  # [num_cols, bert_dim]
            elif self.aggregation_strategy == "max":
                pooled = hidden_states.max(dim=1)[0]  # [num_cols, bert_dim]
            else:
                raise ValueError(
                    f"Unknown aggregation strategy: {self.aggregation_strategy}"
                )

            # Project to target dimension
            projected = self.projection(pooled)  # [num_cols, target_dim]

            # Store as buffer (not a parameter, but persistent)
            self.register_buffer("column_embeddings", projected)

    def _process_column_name(self, column_name: str) -> str:
        """Process a column name according to tokenization strategy.

        Args:
            column_name: Raw column name

        Returns:
            Processed text for BERT encoding
        """
        if self.tokenization_strategy == "as_is":
            return column_name.lower()

        elif self.tokenization_strategy == "split_underscore":
            # Split on underscores
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
                    # If regex didn't match, use the part as-is
                    camel_tokens = [part]

                all_tokens.extend(camel_tokens)

            return " ".join(token.lower() for token in all_tokens if token)

        else:
            raise ValueError(
                f"Unknown tokenization strategy: {self.tokenization_strategy}"
            )

    def get_embeddings(self) -> torch.Tensor:
        """Get column embeddings.

        Returns:
            Column embeddings tensor [num_columns, target_dim]
        """
        if self.use_bert:
            return self.column_embeddings
        else:
            # Generate embeddings on-the-fly for simple approach
            column_embs = self.simple_embedding(
                self.column_indices
            )  # [num_columns, embedding_dim]
            return self.projection(column_embs)  # [num_columns, target_dim]

    def forward(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Create column embeddings for a batch.

        Args:
            batch_size: Batch size
            sequence_length: Sequence length (time dimension)

        Returns:
            Column embeddings shaped [batch_size, num_columns, sequence_length]
            Ready to be concatenated with value/mask channels
        """
        # Get pre-computed embeddings [num_columns, target_dim]
        column_embs = self.get_embeddings()

        # Expand to match batch and time dimensions
        # [num_columns, target_dim] -> [1, num_columns, 1, target_dim]
        column_embs = column_embs.unsqueeze(0).unsqueeze(2)

        # Repeat for batch and time dimensions
        # [1, num_columns, 1, target_dim] -> [batch_size, num_columns, sequence_length, target_dim]
        column_embs = column_embs.repeat(batch_size, 1, sequence_length, 1)

        # Transpose to [batch_size, num_columns, target_dim, sequence_length]
        # This matches the expected [B, C, D, T] format where D=target_dim (usually 1)
        column_embs = column_embs.permute(0, 1, 3, 2)

        # Squeeze out the target_dim if it's 1 to get [B, C, T]
        if self.target_dim == 1:
            column_embs = column_embs.squeeze(2)

        return column_embs


def create_column_embedding(
    column_names: list[str],
    target_dim: int = 1,
    use_bert: bool = False,
    **kwargs: Any,
) -> ColumnEmbedding:
    """Factory function to create a ColumnEmbedding instance.

    Args:
        column_names: List of column names
        target_dim: Target embedding dimension (usually 1 to match value/mask)
        use_bert: Whether to use BERT embeddings (heavy) or simple embeddings (light)
        **kwargs: Additional arguments passed to ColumnEmbedding

    Returns:
        Configured ColumnEmbedding instance
    """
    return ColumnEmbedding(
        column_names=column_names,
        target_dim=target_dim,
        use_bert=use_bert,
        **kwargs,
    )


# Example usage for common datasets
ETTH1_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

if __name__ == "__main__":
    # Test the column embedding module
    column_emb = create_column_embedding(
        column_names=ETTH1_COLUMNS,
        target_dim=1,
    )

    # Test forward pass
    batch_size, seq_len = 32, 96
    embeddings = column_emb(batch_size, seq_len)
    print(f"Column embeddings shape: {embeddings.shape}")
    print(f"Expected shape: [{batch_size}, {len(ETTH1_COLUMNS)}, {seq_len}]")

    # Test tokenization strategies
    test_names = ["user_account_balance", "getUserID", "OT", "temp_sensor_1"]
    for strategy in ["as_is", "split_underscore", "split_underscore_camel"]:
        print(f"\nTokenization strategy: {strategy}")
        emb = ColumnEmbedding(test_names, target_dim=1, tokenization_strategy=strategy)
        for i, name in enumerate(test_names):
            processed = emb._process_column_name(name)
            print(f"  {name} -> {processed}")
