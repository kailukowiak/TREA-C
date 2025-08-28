"""Pre-training PatchDuET with self-supervised learning objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssl_objectives import SSLObjectives
from .variable_feature_patch_duet import VariableFeaturePatchDuET


class PretrainPatchDuET(VariableFeaturePatchDuET):
    """VariableFeaturePatchDuET extended with self-supervised learning for
    pre-training."""

    def __init__(
        self,
        max_numeric_features: int,
        max_categorical_features: int,
        num_classes: int,
        # SSL-specific parameters
        ssl_objectives: dict[str, bool] | None = None,
        mask_ratio: float = 0.15,
        temporal_shuffle_ratio: float = 0.3,
        contrastive_temperature: float = 0.1,
        augmentation_strength: float = 0.1,
        lambda_masked: float = 1.0,
        lambda_temporal: float = 0.5,
        lambda_contrastive: float = 0.3,
        lambda_supervised: float = 0.1,  # Weight for supervised loss
        # Base model parameters
        **kwargs,
    ):
        """Initialize Pre-training PatchDuET.

        Args:
            max_numeric_features: Maximum number of numeric features
            max_categorical_features: Maximum number of categorical features
            num_classes: Number of classes (for supervised component)
            ssl_objectives: Dict specifying which SSL objectives to use
            mask_ratio: Ratio of patches to mask for MPP
            temporal_shuffle_ratio: Ratio for temporal order prediction
            contrastive_temperature: Temperature for contrastive learning
            augmentation_strength: Strength of data augmentation
            lambda_masked: Weight for masked patch prediction loss
            lambda_temporal: Weight for temporal order prediction loss
            lambda_contrastive: Weight for contrastive loss
            lambda_supervised: Weight for supervised loss during pre-training
            **kwargs: Additional arguments for base model
        """
        super().__init__(
            max_numeric_features=max_numeric_features,
            max_categorical_features=max_categorical_features,
            num_classes=num_classes,
            **kwargs,
        )

        # Set default dataset schema to enable forward passes
        self.set_dataset_schema(
            numeric_features=max_numeric_features,
            categorical_features=max_categorical_features,
        )

        # SSL configuration
        if ssl_objectives is None:
            ssl_objectives = {
                "masked_patch": True,
                "temporal_order": True,
                "contrastive": True,
            }

        self.ssl_objectives_config = ssl_objectives
        self.lambda_supervised = lambda_supervised

        # Initialize SSL objectives
        self.ssl_objectives = SSLObjectives(
            mask_ratio=mask_ratio,
            temporal_shuffle_ratio=temporal_shuffle_ratio,
            contrastive_temperature=contrastive_temperature,
            augmentation_strength=augmentation_strength,
            lambda_masked=lambda_masked,
            lambda_temporal=lambda_temporal,
            lambda_contrastive=lambda_contrastive,
        )

        # SSL prediction heads
        self.ssl_heads = nn.ModuleDict()

        # Calculate patch dimensions (this should match the base class calculation)
        total_features = max_numeric_features + max_categorical_features
        base_channels = 2  # value + nan_mask
        if self.use_feature_masks:
            base_channels += 1
        if self.use_column_embeddings:
            base_channels += 1

        # Patch dimension calculation must match base class
        patch_dim = self.patch_len * (base_channels * total_features)
        self.expected_patch_dim = patch_dim

        if ssl_objectives.get("masked_patch", False):
            # Reconstruction head: predict original patch values
            self.ssl_heads["reconstruction"] = nn.Sequential(
                nn.Linear(self.hparams.d_model, self.hparams.d_model),
                nn.ReLU(),
                nn.Linear(self.hparams.d_model, patch_dim),
            )

        if ssl_objectives.get("temporal_order", False):
            # Temporal order head: predict patch ordering
            # (dynamic based on actual patches)
            # We'll adjust this during forward pass based on actual number of patches
            self.ssl_heads["temporal_order"] = nn.Sequential(
                nn.Linear(self.hparams.d_model, self.hparams.d_model),
                nn.ReLU(),
                nn.Linear(
                    self.hparams.d_model, self.hparams.d_model
                ),  # Will project to num_patches in forward
            )

        if ssl_objectives.get("contrastive", False):
            # Contrastive projection head
            self.ssl_heads["contrastive"] = nn.Sequential(
                nn.Linear(self.hparams.d_model, self.hparams.d_model),
                nn.ReLU(),
                nn.Linear(self.hparams.d_model, 128),  # Projection dimension
            )

    def create_patches_with_features(
        self,
        x_num: torch.Tensor,
        feature_mask: torch.Tensor,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create patches with all feature processing for SSL.

        Args:
            x_num: Numeric features [B, max_features, T]
            feature_mask: Feature mask [B, max_features, T]
            x_cat: Categorical features (optional)

        Returns:
            Processed patches [B, num_patches, patch_dim]
        """
        B, _, T = x_num.shape

        # Use the same feature padding approach as the base class forward method
        x_padded, feature_mask_padded = self.pad_features_to_unified_space(x_num, x_cat)
        # x_padded: [B, max_total_features, T]
        # feature_mask_padded: [B, max_total_features, T]

        # Handle NaNs on the padded space
        m_nan = torch.isnan(x_padded).float()
        x_val = torch.nan_to_num(x_padded, nan=0.0)

        # Combine channels
        channels_to_cat = [x_val, m_nan]

        if self.use_feature_masks:
            channels_to_cat.append(feature_mask_padded)

        if self.use_column_embeddings and self.column_embedder is not None:
            col_emb = self.column_embedder(B, T)
            channels_to_cat.append(col_emb)

        x_processed = torch.cat(channels_to_cat, dim=1)

        # Create patches using base class method
        patches = self.create_patches(x_processed)

        return patches

    def get_global_embedding(self, patches: torch.Tensor) -> torch.Tensor:
        """Get global embedding from patches for contrastive learning.

        Args:
            patches: Input patches [B, num_patches, patch_dim]

        Returns:
            Global embeddings [B, d_model]
        """
        B, num_patches, _ = patches.shape

        # Patch embedding
        patch_embeddings = self.patch_embedding(patches)

        # Add positional encoding
        pos_enc = self.get_positional_encoding(num_patches)
        patch_embeddings = patch_embeddings + pos_enc

        # Transformer
        transformer_out = self.transformer(patch_embeddings)

        # Global pooling
        global_embedding = transformer_out.mean(dim=1)

        return global_embedding

    def forward_ssl(
        self,
        x_num: torch.Tensor,
        feature_mask: torch.Tensor,
        x_cat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for SSL objectives.

        Args:
            x_num: Numeric features [B, max_features, T]
            feature_mask: Feature mask [B, max_features, T]
            x_cat: Categorical features (optional)

        Returns:
            Dictionary of SSL predictions
        """
        B, C, T = x_num.shape

        # Create patches
        patches = self.create_patches_with_features(x_num, feature_mask, x_cat)
        num_patches = patches.shape[1]

        ssl_outputs = {}

        # Masked Patch Prediction
        if (
            "masked_patch" in self.ssl_objectives_config
            and self.ssl_objectives_config["masked_patch"]
        ):
            # Create mask
            mask = self.ssl_objectives.masked_predictor.create_mask(
                num_patches, x_num.device
            )
            masked_patches, targets = (
                self.ssl_objectives.masked_predictor.apply_masking(patches, mask)
            )

            # Forward through transformer
            patch_embeddings = self.patch_embedding(masked_patches)
            pos_enc = self.get_positional_encoding(num_patches)
            patch_embeddings = patch_embeddings + pos_enc
            transformer_out = self.transformer(patch_embeddings)

            # Reconstruction
            reconstructed = self.ssl_heads["reconstruction"](transformer_out)

            ssl_outputs["masked_patch"] = {
                "predictions": reconstructed,
                "targets": targets,
                "mask": mask,
            }

        # Temporal Order Prediction
        if (
            "temporal_order" in self.ssl_objectives_config
            and self.ssl_objectives_config["temporal_order"]
        ):
            shuffled_patches, order_targets = (
                self.ssl_objectives.temporal_predictor.create_shuffled_sequence(patches)
            )

            # Forward through transformer
            patch_embeddings = self.patch_embedding(shuffled_patches)
            pos_enc = self.get_positional_encoding(num_patches)
            patch_embeddings = patch_embeddings + pos_enc
            transformer_out = self.transformer(patch_embeddings)

            # Temporal order prediction with dynamic projection
            temporal_features = self.ssl_heads["temporal_order"](
                transformer_out.mean(dim=1)
            )

            # Dynamic projection to num_patches
            if (
                not hasattr(self, "temporal_projection")
                or self.temporal_projection.out_features != num_patches
            ):
                self.temporal_projection = nn.Linear(
                    self.hparams.d_model, num_patches
                ).to(temporal_features.device)
                # Initialize the projection weights properly
                torch.nn.init.xavier_uniform_(self.temporal_projection.weight)
                torch.nn.init.zeros_(self.temporal_projection.bias)

            order_logits = self.temporal_projection(temporal_features)

            # Debug: Check for NaN in temporal components
            if torch.isnan(temporal_features).any() or torch.isnan(order_logits).any():
                print("NaN in temporal order prediction!")
                print(
                    "  temporal_features has NaN: "
                    f"{torch.isnan(temporal_features).any()}"
                )
                print(f"  order_logits has NaN: {torch.isnan(order_logits).any()}")
                print(f"  num_patches: {num_patches}")
                order_logits = torch.zeros_like(order_logits)

            ssl_outputs["temporal_order"] = {
                "predictions": order_logits,
                "targets": order_targets,
            }

        # Contrastive Learning
        if (
            "contrastive" in self.ssl_objectives_config
            and self.ssl_objectives_config["contrastive"]
        ):
            # Original embedding
            orig_embedding = self.get_global_embedding(patches)
            orig_projection = self.ssl_heads["contrastive"](orig_embedding)

            # Augmented embedding
            x_aug = self.ssl_objectives.contrastive_learner.augment_time_series(x_num)
            aug_patches = self.create_patches_with_features(x_aug, feature_mask, x_cat)
            aug_embedding = self.get_global_embedding(aug_patches)
            aug_projection = self.ssl_heads["contrastive"](aug_embedding)

            ssl_outputs["contrastive"] = {
                "original": orig_projection,
                "augmented": aug_projection,
            }

        return ssl_outputs

    def compute_ssl_losses(
        self, ssl_outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute SSL losses from SSL outputs.

        Args:
            ssl_outputs: Outputs from forward_ssl

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Masked Patch Prediction Loss
        if "masked_patch" in ssl_outputs:
            outputs = ssl_outputs["masked_patch"]
            predictions = outputs["predictions"]
            targets = outputs["targets"]
            mask = outputs["mask"]

            # Reshape for loss computation
            B, num_patches, patch_dim = predictions.shape
            predictions_flat = predictions.view(B, num_patches, -1)
            targets_flat = targets.view(B, num_patches, -1)

            # Apply mask
            mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand_as(predictions_flat)

            if mask_expanded.sum() > 0:  # Ensure we have masked patches
                pred_masked = predictions_flat[mask_expanded]
                targ_masked = targets_flat[mask_expanded]

                # Check for NaN/Inf values
                if (
                    torch.isnan(pred_masked).any()
                    or torch.isnan(targ_masked).any()
                    or torch.isinf(pred_masked).any()
                    or torch.isinf(targ_masked).any()
                ):
                    reconstruction_loss = torch.tensor(0.0, device=predictions.device)
                else:
                    reconstruction_loss = F.mse_loss(pred_masked, targ_masked)
                    # Clamp to prevent explosion
                    reconstruction_loss = torch.clamp(reconstruction_loss, 0, 100.0)

                losses["reconstruction"] = (
                    reconstruction_loss * self.ssl_objectives.lambda_masked
                )

        # Temporal Order Prediction Loss
        if "temporal_order" in ssl_outputs:
            outputs = ssl_outputs["temporal_order"]
            predictions = outputs["predictions"]
            targets = outputs["targets"]

            # Check for NaN/Inf values
            if (
                torch.isnan(predictions).any()
                or torch.isnan(targets).any()
                or torch.isinf(predictions).any()
                or torch.isinf(targets).any()
            ):
                temporal_loss = torch.tensor(0.0, device=predictions.device)
            else:
                temporal_loss = F.mse_loss(predictions, targets.float())
                # Clamp to prevent explosion
                temporal_loss = torch.clamp(temporal_loss, 0, 100.0)

            losses["temporal_order"] = (
                temporal_loss * self.ssl_objectives.lambda_temporal
            )

        # Contrastive Loss
        if "contrastive" in ssl_outputs:
            outputs = ssl_outputs["contrastive"]
            original = outputs["original"]
            augmented = outputs["augmented"]

            # Check for NaN/Inf values
            if (
                torch.isnan(original).any()
                or torch.isnan(augmented).any()
                or torch.isinf(original).any()
                or torch.isinf(augmented).any()
            ):
                contrastive_loss = torch.tensor(0.0, device=original.device)
            else:
                contrastive_loss = (
                    self.ssl_objectives.contrastive_learner.contrastive_loss(
                        original, augmented
                    )
                )
                # Clamp to prevent explosion
                contrastive_loss = torch.clamp(contrastive_loss, 0, 10.0)

            losses["contrastive"] = (
                contrastive_loss * self.ssl_objectives.lambda_contrastive
            )

        # Total SSL loss
        if losses:
            total_ssl_loss = sum(losses.values())
            losses["total_ssl"] = total_ssl_loss

        return losses

    def training_step(self, batch, batch_idx):
        """Training step with SSL + supervised learning."""
        x_num = batch["x_num"]
        x_cat = batch.get("x_cat", None)
        y = batch["y"]
        feature_mask = batch["feature_mask"]

        # SSL forward pass
        ssl_outputs = self.forward_ssl(x_num, feature_mask, x_cat)
        ssl_losses = self.compute_ssl_losses(ssl_outputs)

        # Supervised forward pass (standard classification)
        y_hat = self(x_num, x_cat)

        # Handle multi-class datasets in same batch
        supervised_losses = []
        dataset_names = batch["dataset_name"]

        for i, dataset_name in enumerate(dataset_names):
            # Get sample-specific info
            sample_y = y[i : i + 1]
            sample_y_hat = y_hat[i : i + 1]
            sample_num_classes = batch["num_classes"][i].item()

            # Adjust predictions for this dataset's number of classes
            sample_y_hat_adj = sample_y_hat[:, :sample_num_classes]

            if sample_y.ndim == 2:
                sample_y = torch.argmax(sample_y, dim=1)
            sample_y = sample_y.long()

            if sample_y.max() < sample_num_classes:  # Valid label
                loss = self.loss_fn(sample_y_hat_adj, sample_y)
                supervised_losses.append(loss)

        # Average supervised loss
        if supervised_losses:
            supervised_loss = torch.stack(supervised_losses).mean()
        else:
            supervised_loss = torch.tensor(0.0, device=x_num.device)

        # Total loss
        total_ssl_loss = ssl_losses.get(
            "total_ssl", torch.tensor(0.0, device=x_num.device)
        )
        total_loss = total_ssl_loss + self.lambda_supervised * supervised_loss

        # Debug: Check for NaN in individual components
        if (
            torch.isnan(total_ssl_loss)
            or torch.isnan(supervised_loss)
            or torch.isnan(total_loss)
        ):
            print("NaN detected!")
            print(f"  SSL loss: {total_ssl_loss}")
            print(f"  Supervised loss: {supervised_loss}")
            print(f"  Total loss: {total_loss}")
            print(f"  SSL components: {[(k, v.item()) for k, v in ssl_losses.items()]}")

            # Return a small loss to prevent crash
            total_loss = torch.tensor(1.0, device=x_num.device, requires_grad=True)

        # Logging
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/ssl_loss", total_ssl_loss)
        self.log("train/supervised_loss", supervised_loss)

        for loss_name, loss_value in ssl_losses.items():
            if loss_name != "total_ssl":
                self.log(f"train/{loss_name}_loss", loss_value)

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step (can handle multiple validation loaders)."""
        x_num = batch["x_num"]
        x_cat = batch.get("x_cat", None)
        y = batch["y"]

        # Standard supervised validation
        y_hat = self(x_num, x_cat)

        if y.ndim == 2:
            y = torch.argmax(y, dim=1)
        y = y.long()

        # Get dataset info
        dataset_name = batch["dataset_name"][
            0
        ]  # Assume same dataset in validation batch
        num_classes = batch["num_classes"][0].item()

        # Adjust predictions for this dataset's number of classes
        y_hat_adj = y_hat[:, :num_classes]

        loss = self.loss_fn(y_hat_adj, y)

        # Log with dataset prefix
        self.log(f"val/{dataset_name}_loss", loss, add_dataloader_idx=False)

        return loss

    @classmethod
    def from_pretrained(
        cls, pretrained_path: str, num_classes: int, freeze_encoder: bool = True
    ) -> "PretrainPatchDuET":
        """Load pre-trained model for fine-tuning.

        Args:
            pretrained_path: Path to pre-trained checkpoint
            num_classes: Number of classes for fine-tuning task
            freeze_encoder: Whether to freeze encoder layers

        Returns:
            Model loaded from pre-trained weights
        """
        # Load checkpoint
        checkpoint = torch.load(pretrained_path, map_location="cpu")

        # Extract hyperparameters
        hparams = checkpoint.get("hyper_parameters", {})

        # Create model
        model = cls(
            max_numeric_features=hparams["max_numeric_features"],
            max_categorical_features=hparams["max_categorical_features"],
            num_classes=num_classes,  # Use target number of classes
            **{k: v for k, v in hparams.items() if k not in ["num_classes"]},
        )

        # Load state dict (skip SSL heads and mismatched classification head)
        state_dict = checkpoint["state_dict"]

        # Filter out SSL heads and classification head
        encoder_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("ssl_heads.") and not k.startswith("head.")
        }

        # Load encoder weights
        model.load_state_dict(encoder_state_dict, strict=False)

        # Freeze encoder if requested
        if freeze_encoder:
            for name, param in model.named_parameters():
                if not name.startswith("head."):
                    param.requires_grad = False

        print(f"Loaded pre-trained model from {pretrained_path}")
        print(f"Frozen encoder: {freeze_encoder}")

        return model


if __name__ == "__main__":
    # Test pre-training model
    print("Testing PretrainPatchDuET...")

    # Create model
    model = PretrainPatchDuET(
        max_numeric_features=12,
        max_categorical_features=0,
        num_classes=6,
        ssl_objectives={
            "masked_patch": True,
            "temporal_order": True,
            "contrastive": True,
        },
        d_model=64,
        n_head=4,
        num_layers=2,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    B, C, T = 4, 12, 96
    x_num = torch.randn(B, C, T)
    feature_mask = torch.ones(B, C, T)

    # Add some NaNs
    x_num[0, :2, :10] = float("nan")

    # SSL forward pass
    ssl_outputs = model.forward_ssl(x_num, feature_mask)
    ssl_losses = model.compute_ssl_losses(ssl_outputs)

    print(f"SSL outputs keys: {list(ssl_outputs.keys())}")
    print(f"SSL losses: {[(k, v.item()) for k, v in ssl_losses.items()]}")

    # Standard forward pass
    output = model(x_num)
    print(f"Classification output: {output.shape}")

    print("✅ PretrainPatchDuET test complete!")
