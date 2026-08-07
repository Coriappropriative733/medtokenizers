"""Training configuration classes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossConfig:
    """Configuration for multi-loss training.

    Supports L1 reconstruction, quantization, VGG perceptual, Gram style,
    and Laplacian smoothness losses with stage-based scheduling.
    """

    l1_weight: float = 1.0
    quant_weight: float = 0.0
    vgg_weight: float = 0.0
    gram_weight: float = 0.0
    laplacian_weight: float = 0.0

    stage1_epochs: int = 0
    stage1_vgg_weight: float = 0.0
    stage1_gram_weight: float = 0.0

    stage2_vgg_weight: float = 0.0
    stage2_gram_weight: float = 0.0
    stage2_warmup_epochs: int = 0

    laplacian_start_epoch: int = 0

    def compute_stage_weights(self, epoch: int) -> tuple[float, float, int]:
        """Compute VGG/Gram weights and current stage for given epoch.

        Args:
            epoch: Current epoch number

        Returns:
            (vgg_weight, gram_weight, stage)
        """
        if epoch < self.stage1_epochs:
            return self.stage1_vgg_weight, self.stage1_gram_weight, 1

        if self.stage2_warmup_epochs <= 0:
            return self.stage2_vgg_weight, self.stage2_gram_weight, 2

        progress = min((epoch - self.stage1_epochs) / self.stage2_warmup_epochs, 1.0)
        vgg = self.stage1_vgg_weight + progress * (
            self.stage2_vgg_weight - self.stage1_vgg_weight
        )
        gram = self.stage1_gram_weight + progress * (
            self.stage2_gram_weight - self.stage1_gram_weight
        )
        return vgg, gram, 2

    @property
    def uses_vgg(self) -> bool:
        """Whether VGG loss is enabled in any stage."""
        return (
            self.vgg_weight > 0
            or self.stage1_vgg_weight > 0
            or self.stage2_vgg_weight > 0
        )

    @property
    def uses_gram(self) -> bool:
        """Whether Gram loss is enabled in any stage."""
        return (
            self.gram_weight > 0
            or self.stage1_gram_weight > 0
            or self.stage2_gram_weight > 0
        )

    @property
    def uses_laplacian(self) -> bool:
        """Whether Laplacian loss is enabled."""
        return self.laplacian_weight > 0
