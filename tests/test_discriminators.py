"""Tests for discriminator networks."""

import torch

from medtokenizers.training.discriminators import MultiScale, PatchGAN, StyleGAN


class TestPatchGAN:
    """Tests for PatchGAN discriminator."""

    def test_2d_forward(self):
        disc = PatchGAN(dim=2, input_channels=1, ndf=32, n_layers=2)
        x = torch.randn(2, 1, 64, 64)

        output = disc(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 1
        assert output.ndim == 4  # Batch, channels, H, W

    def test_3d_forward(self):
        disc = PatchGAN(dim=3, input_channels=1, ndf=16, n_layers=2)
        x = torch.randn(1, 1, 16, 32, 32)

        output = disc(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 1
        assert output.ndim == 5  # Batch, channels, D, H, W

    def test_multichannel_input(self):
        disc = PatchGAN(dim=2, input_channels=3, ndf=32)
        x = torch.randn(2, 3, 64, 64)

        output = disc(x)

        assert output.shape[0] == 2

    def test_with_spectral_norm(self):
        disc = PatchGAN(dim=2, input_channels=1, use_spectral_norm=True)
        x = torch.randn(1, 1, 64, 64)

        output = disc(x)

        assert output.ndim == 4

    def test_without_spectral_norm(self):
        disc = PatchGAN(dim=2, input_channels=1, use_spectral_norm=False)
        x = torch.randn(1, 1, 64, 64)

        output = disc(x)

        assert output.ndim == 4

    def test_different_layer_depths(self):
        for n_layers in [2, 3, 4]:
            disc = PatchGAN(dim=2, input_channels=1, n_layers=n_layers, ndf=16)
            x = torch.randn(1, 1, 64, 64)

            output = disc(x)

            assert output.shape[0] == 1


class TestMultiScale:
    """Tests for MultiScale discriminator."""

    def test_returns_multiple_scales(self):
        disc = MultiScale(dim=2, input_channels=1, num_scales=3, ndf=16, n_layers=2)
        x = torch.randn(1, 1, 64, 64)

        outputs = disc(x)

        assert isinstance(outputs, list)
        assert len(outputs) == 3
        # Each output should be a tensor
        for out in outputs:
            assert isinstance(out, torch.Tensor)
            assert out.shape[0] == 1  # batch size

    def test_output_sizes_decrease(self):
        disc = MultiScale(dim=2, input_channels=1, num_scales=3, ndf=16)
        x = torch.randn(1, 1, 128, 128)

        outputs = disc(x)

        # Each scale should have different spatial dimensions
        sizes = [out.shape[-1] for out in outputs]
        assert sizes[0] > sizes[1] > sizes[2]

    def test_3d_multiscale(self):
        disc = MultiScale(dim=3, input_channels=1, num_scales=2, ndf=8, n_layers=2)
        x = torch.randn(1, 1, 32, 32, 32)

        outputs = disc(x)

        assert len(outputs) == 2
        assert all(out.ndim == 5 for out in outputs)

    def test_single_scale(self):
        disc = MultiScale(dim=2, input_channels=1, num_scales=1, ndf=16)
        x = torch.randn(1, 1, 64, 64)

        outputs = disc(x)

        assert len(outputs) == 1


class TestStyleGAN:
    """Tests for StyleGAN discriminator."""

    def test_2d_forward(self):
        disc = StyleGAN(dim=2, input_channels=1, base_channels=32, num_blocks=3)
        x = torch.randn(2, 1, 64, 64)

        output = disc(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 1

    def test_3d_forward(self):
        disc = StyleGAN(dim=3, input_channels=1, base_channels=16, num_blocks=2)
        x = torch.randn(1, 1, 16, 32, 32)

        output = disc(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 1

    def test_channel_progression(self):
        disc = StyleGAN(
            dim=2, input_channels=1, base_channels=32, max_channels=128, num_blocks=4
        )
        x = torch.randn(1, 1, 64, 64)

        output = disc(x)

        assert output.shape[0] == 1

    def test_different_num_blocks(self):
        for num_blocks in [2, 3, 4]:
            disc = StyleGAN(
                dim=2, input_channels=1, base_channels=16, num_blocks=num_blocks
            )
            x = torch.randn(1, 1, 64, 64)

            output = disc(x)

            assert output.shape[0] == 1


class TestDiscriminatorEdgeCases:
    """Edge case tests for all discriminators."""

    def test_batch_size_one(self):
        disc = PatchGAN(dim=2, input_channels=1, ndf=16)
        x = torch.randn(1, 1, 64, 64)

        output = disc(x)

        assert output.shape[0] == 1

    def test_small_spatial_size(self):
        disc = PatchGAN(dim=2, input_channels=1, ndf=16, n_layers=2)
        x = torch.randn(1, 1, 32, 32)

        output = disc(x)

        assert output.ndim == 4

    def test_gradient_flow(self):
        disc = PatchGAN(dim=2, input_channels=1, ndf=16)
        x = torch.randn(1, 1, 64, 64, requires_grad=True)

        output = disc(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_reproducibility(self):
        torch.manual_seed(42)
        disc1 = PatchGAN(dim=2, input_channels=1, ndf=16)
        x = torch.randn(1, 1, 64, 64)
        out1 = disc1(x)

        torch.manual_seed(42)
        disc2 = PatchGAN(dim=2, input_channels=1, ndf=16)
        out2 = disc2(x)

        assert torch.allclose(out1, out2)
