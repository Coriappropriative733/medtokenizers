"""Tests for inference utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from medtokenizers.inference import (
    load_indices,
    load_latents,
    save_indices,
    save_latents,
)


class TestLatentIO:
    """Test save_latents and load_latents functionality."""

    def test_save_and_load_tensor(self):
        """Test saving and loading torch tensors (default float16)."""
        latents = torch.randn(2, 4, 8, 8, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "latents"
            save_latents(latents, save_path)  # default is float16

            loaded = load_latents(save_path.with_suffix(".npz"))

            # float16 has ~3 decimal digits of precision
            assert torch.allclose(loaded, latents, atol=1e-3)
            assert loaded.shape == latents.shape

    def test_save_and_load_numpy(self):
        """Test saving and loading numpy arrays (default float16)."""
        latents = np.random.randn(2, 4, 8, 8, 8).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "latents.npz"
            save_latents(latents, save_path)  # default is float16

            loaded = load_latents(save_path)

            # float16 has ~3 decimal digits of precision but relative error can be higher
            # for values > 1, so use rtol for proper comparison
            assert np.allclose(loaded.numpy(), latents, rtol=1e-2, atol=1e-3)
            assert loaded.shape == tuple(latents.shape)

    def test_save_creates_parent_dirs(self):
        """Test that save_latents creates parent directories."""
        latents = torch.randn(1, 4, 4, 4, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dirs" / "latents"
            save_latents(latents, save_path)

            # Check file was created
            assert save_path.with_suffix(".npz").exists()

    def test_load_with_device(self):
        """Test loading latents to a specific device."""
        latents = torch.randn(1, 4, 4, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "latents.npz"
            save_latents(latents, save_path)

            loaded = load_latents(save_path, device="cpu")

            assert loaded.device.type == "cpu"
            # float16 has ~3 decimal digits of precision
            assert torch.allclose(loaded, latents, atol=1e-3)

    def test_string_paths(self):
        """Test that string paths work as well as Path objects."""
        latents = torch.randn(1, 4, 4, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "latents")
            save_latents(latents, save_path)

            loaded = load_latents(save_path + ".npz")

            # float16 has ~3 decimal digits of precision
            assert torch.allclose(loaded, latents, atol=1e-3)

    def test_save_float16(self):
        """Test saving latents as float16."""
        latents = torch.randn(2, 4, 8, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "latents"
            save_latents(latents, save_path, dtype="float16")

            # Check saved dtype
            data = np.load(save_path.with_suffix(".npz"))
            assert data["latents"].dtype == np.float16

            # Verify loaded values are close (float16 has limited precision)
            loaded = load_latents(save_path.with_suffix(".npz"))
            assert torch.allclose(loaded, latents, atol=1e-3)

    def test_save_float32(self):
        """Test saving latents as float32."""
        latents = torch.randn(2, 4, 8, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "latents"
            save_latents(latents, save_path, dtype="float32")

            # Check saved dtype
            data = np.load(save_path.with_suffix(".npz"))
            assert data["latents"].dtype == np.float32


class TestIndicesIO:
    """Test save_indices and load_indices functionality."""

    def test_save_and_load_tensor(self):
        """Test saving and loading torch tensors."""
        indices = torch.randint(0, 512, (2, 8, 8, 8), dtype=torch.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "indices"
            save_indices(indices, save_path)

            loaded = load_indices(save_path.with_suffix(".npz"))

            assert torch.equal(loaded, indices.to(torch.int64))
            assert loaded.shape == indices.shape

    def test_save_and_load_numpy(self):
        """Test saving and loading numpy arrays."""
        indices = np.random.randint(0, 512, (2, 8, 8, 8), dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "indices.npz"
            save_indices(indices, save_path)

            loaded = load_indices(save_path)

            assert np.array_equal(loaded.numpy(), indices)
            assert loaded.shape == tuple(indices.shape)

    def test_save_int16(self):
        """Test saving indices as int16 (default)."""
        indices = torch.randint(0, 512, (2, 8, 8, 8), dtype=torch.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "indices"
            save_indices(indices, save_path, dtype="int16")

            # Check saved dtype
            data = np.load(save_path.with_suffix(".npz"))
            assert data["indices"].dtype == np.int16

    def test_save_int32(self):
        """Test saving indices as int32."""
        indices = torch.randint(0, 50000, (2, 8, 8, 8), dtype=torch.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "indices"
            save_indices(indices, save_path, dtype="int32")

            # Check saved dtype
            data = np.load(save_path.with_suffix(".npz"))
            assert data["indices"].dtype == np.int32

    def test_load_with_device(self):
        """Test loading indices to a specific device."""
        indices = torch.randint(0, 512, (1, 4, 4, 4), dtype=torch.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "indices.npz"
            save_indices(indices, save_path)

            loaded = load_indices(save_path, device="cpu")

            assert loaded.device.type == "cpu"

    def test_load_with_dtype(self):
        """Test loading indices with specific dtype."""
        indices = torch.randint(0, 512, (1, 4, 4, 4), dtype=torch.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "indices.npz"
            save_indices(indices, save_path)

            loaded = load_indices(save_path, dtype=torch.int32)

            assert loaded.dtype == torch.int32

    def test_int16_range(self):
        """Test that int16 correctly stores values in valid range."""
        # Max int16 is 32767
        indices = torch.tensor([0, 100, 512, 4096, 32000], dtype=torch.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "indices"
            save_indices(indices, save_path, dtype="int16")

            loaded = load_indices(save_path.with_suffix(".npz"))
            assert torch.equal(loaded, indices)

    def test_save_indices_overflow_raises(self):
        """Test that overflow beyond int16 raises an error."""
        indices = torch.tensor([0, 40000], dtype=torch.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "indices"
            with pytest.raises(ValueError, match="int16"):
                save_indices(indices, save_path)

    def test_save_indices_negative_raises(self):
        """Test that negative indices are rejected."""
        indices = torch.tensor([-1, 0, 1], dtype=torch.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "indices"
            with pytest.raises(ValueError, match="non-negative"):
                save_indices(indices, save_path)

    def test_creates_parent_dirs(self):
        """Test that save_indices creates parent directories."""
        indices = torch.randint(0, 512, (1, 4, 4, 4), dtype=torch.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dirs" / "indices"
            save_indices(indices, save_path)

            assert save_path.with_suffix(".npz").exists()
