# Security Policy

## Supported Versions

`medtokenizers` is currently released as a `0.1.x` preview. Security fixes are
applied to the latest released version.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities **privately**. Do **not** open a public
GitHub issue for security problems.

Email **liamchalcroft@gmail.com** with:

- a description of the vulnerability and its impact,
- steps to reproduce or a proof of concept, and
- any suggested remediation.

You can expect an initial acknowledgement within a few business days. Once the
issue is confirmed and a fix is available, we will coordinate disclosure.

## Loading Untrusted Models

`medtokenizers` loads model checkpoints with PyTorch. As of this release, all
checkpoint loading uses `torch.load(..., weights_only=True)` to mitigate
arbitrary code execution from maliciously crafted checkpoints.

Even so, **only load models and weights from sources you trust.** Treat
third-party `from_pretrained` / `load_tokenizer` targets (including HuggingFace
Hub repositories) as untrusted input unless you have verified their provenance.
