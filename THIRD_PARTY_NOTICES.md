# Third-Party Notices

`medtokenizers` is released under the MIT License (Copyright (c) 2026 Liam
Chalcroft). It incorporates code derived from the third-party open-source
projects listed below. Each derived source file carries an SPDX attribution
header pointing back to this document. The original license terms of each
upstream project continue to apply to the derived portions. In particular, the
Apache-2.0-licensed portions (NVIDIA Cosmos-Tokenizer, MONAI/MAISI) remain
governed by the Apache License, Version 2.0, and the MIT-licensed portions by
their respective MIT terms.

---

## CompVis latent-diffusion / Stable Diffusion / taming-transformers

- **Upstream:** https://github.com/CompVis/latent-diffusion
  (the same `diffusionmodules/model.py` also ships in
  https://github.com/CompVis/stable-diffusion and
  https://github.com/CompVis/taming-transformers)
- **License:** MIT
- **Copyright:** (c) 2021-2022 CompVis and contributors
- **Derived files:**
  - `src/medtokenizers/modules/layers.py`
- **What is derived:** The `Encoder`, `Decoder`, `ResnetBlock`, `AttnBlock`,
  `Upsample`, and `Downsample` classes and the `nonlinearity` (swish) and
  `Normalize` (GroupNorm, eps=1e-6) helpers follow the CompVis autoencoder
  design, including the `nin_shortcut` / `norm_out` / `conv_out` naming, the
  `mid.block_1 / attn_1 / block_2` bottleneck, the `(0, 1, 0, 1)` asymmetric
  downsampling pad, and the `in_ch_mult = (1,) + tuple(...)` channel schedule.
- **Local modifications:** Generalized to both 2D and 3D via a `get_conv`
  factory and channels-last memory formats; attention reimplemented with
  `torch.nn.functional.scaled_dot_product_attention`; optional gradient
  checkpointing; MAISI-compatibility switches (`use_encoder_mid`,
  `decoder_blocks_per_stage`, `use_output_nonlinearity`).

---

## vector-quantize-pytorch (lucidrains)

- **Upstream:** https://github.com/lucidrains/vector-quantize-pytorch
- **License:** MIT
- **Copyright:** (c) 2020 Phil Wang
- **Additional reference:** The Finite Scalar Quantization formulation derives
  from the official JAX snippet in Mentzer et al., "Finite Scalar Quantization:
  VQ-VAE Made Simple" (arXiv:2309.15505, Appendix A.1).
- **Derived files:**
  - `src/medtokenizers/modules/quant.py`
- **What is derived:**
  - `FSQuantizer`: the `bound` / `quantize` / `codes_to_indices` /
    `indices_to_codes` methods, the `_levels` and `_basis` (cumprod) buffers,
    mixed-radix index encoding, and `round_ste`-based quantization mirror the
    lucidrains FSQ implementation.
  - `ResidualFSQuantizer`: the stacked residual-FSQ scheme.
  - `LFQuantizer`: sign-based lookup-free quantization with a
    straight-through estimator and per-sample-minus-codebook entropy
    regularization, following lucidrains `lookup_free_quantization`.
  - `VectorQuantizer` shares ancestry with the classic VQGAN/taming-transformers
    vector quantizer (also MIT) but has been substantially rewritten.
- **Local modifications:** float32/AMP numerical guards in `bound`,
  `torch.compile`/`fullgraph` safety, DDP-synchronized EMA codebook updates,
  dead-code reset, and chunked nearest-neighbour distance computation.

---

## NVIDIA Cosmos-Tokenizer

- **Upstream:** https://github.com/NVIDIA/Cosmos-Tokenizer
  (`cosmos_tokenizer/modules/patching.py`)
- **License:** Apache License, Version 2.0
- **Copyright:** (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- **Derived files:**
  - `src/medtokenizers/modules/patch.py`
- **What is derived:** The `_WAVELETS` table (identical) and the Haar discrete
  wavelet transform / inverse transform routines (`_dwt_2d`, `_dwt_3d`,
  `_idwt_2d`, `_idwt_3d`) and their `haar` / `rearrange` dispatch used by
  `SpatialCompressor` and `SpatialDecompressor` are adapted from the Cosmos
  `Patcher` / `Patcher3D` / `UnPatcher` / `UnPatcher3D` implementations. (NVIDIA
  Cosmos in turn builds on the MAGVIT/MAGVIT-2 Haar patching idea.)
- **Local additions:** The Voronoi tiling path
  (`_setup_voronoi_boundaries`, `_voronoi_2d`, `_voronoi_3d`,
  `_inverse_voronoi_2d`, `_inverse_voronoi_3d`) is original to this project and
  is not derived from Cosmos.

---

## MONAI / NVIDIA MAISI

- **Upstream:** https://github.com/Project-MONAI/MONAI
  (MAISI lives under `monai/apps/generation/maisi/`)
- **License (code):** Apache License, Version 2.0
- **Copyright:** (c) MONAI Consortium
- **Derived files:**
  - `src/medtokenizers/modules/distributions.py`
  - `src/medtokenizers/networks/nvidia_maisi.py`
- **What is derived:**
  - `distributions.py`: `GaussianDistribution` follows the MAISI-style
    diagonal-Gaussian VAE posterior: logvar clamping to
    `[min_logvar, max_logvar]`, reparameterized sampling, and KL divergence
    against an `N(0, I)` prior. `IdentityDistribution` is original.
  - `nvidia_maisi.py`: `NVIDIAMAISITokenizer` reproduces the MAISI VAE layer
    layout (no encoder mid blocks, `decoder_blocks_per_stage = [2, 2, 0]`,
    separate `quant_conv_mu` / `quant_conv_log_sigma`), and
    `convert_nvidia_weights` / `_convert_resblock` translate published NVIDIA
    MAISI checkpoints into this tokenizer's state dict.
- **Important, model weights:** The NVIDIA MAISI / NV-Generate-MR pretrained
  **weights** are released under the NVIDIA Source Code License (NSCLv1), which
  is separate from and more restrictive than Apache-2.0. This repository's MIT
  license covers only the source code; any MAISI-derived weight files
  that are downloaded or redistributed remain governed by NSCLv1 and must carry
  that license.

---

## BrainWeb Simulated Brain Database (bundled data)

- **Upstream:** https://brainweb.bic.mni.mcgill.ca/brainweb/
  (McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill
  University)
- **License:** None published. BrainWeb states a citation requirement rather
  than a formal licence or an explicit redistribution grant.
- **Bundled file:** `src/medtokenizers/assets/t1w_brainweb.nii.gz`, shipped in
  the wheel and resolved at runtime by `medtokenizers.example_volume_path()`
- **What it is:** Simulated MRI output from a digital brain phantom, derived
  from `t1_icbm_normal_1mm_pn3_rf20.mnc` (normal anatomy, T1, 1 mm isotropic,
  3% noise, 20% intensity non-uniformity). This is not a scan of a human
  subject, so no consent or de-identification question applies.
- **Local modifications:** Converted MINC to NIfTI-1, reoriented to canonical
  RAS, rescaled to uint8 against the 99.9th intensity percentile, gzipped.
- **Scope:** This file is data, not source, and is **not** covered by this
  repository's MIT license. See `src/medtokenizers/assets/README.md` for the
  citations BrainWeb asks for.

---

## Files reviewed and found to be original (no attribution required)

The following files were audited and contain original, from-scratch
implementations. Where a concept or configuration originates upstream, this is
noted, but no upstream *code* is copied, so no attribution header is added:

- `src/medtokenizers/networks/maisi.py`: a thin config subclass of
  `ContinuousTokenizer` that hard-codes MAISI hyperparameters. Original code;
  only the configuration values reflect the MAISI paper.
- `src/medtokenizers/networks/titok.py`: an original implementation of the
  TiTok concept (1D latent tokens, learnable latent/mask tokens) built on
  `torch.nn.TransformerEncoder`. It does not copy code from
  bytedance/1d-tokenizer (which is itself Apache-2.0); only the high-level idea
  is shared.
- `src/medtokenizers/networks/rae.py`: an original implementation of the
  Representation Autoencoder idea (Zheng et al., 2025) using HuggingFace
  `AutoModel` adapters and a custom patch-MLP decoder. No upstream code copied.
- `src/medtokenizers/modules/base.py`: `BaseTokenizer.reconstruct` provides
  Gaussian-weighted sliding-window inference whose behaviour is conceptually
  similar to MONAI's `sliding_window_inference`, but it is an independent
  from-scratch implementation with a different API and weighting; not derived.
