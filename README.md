# FIV1.2

### Work in Progress

Building upon the original single-stream ConvNeXt baseline, FIV1.2 introduces an advanced triple-stream architecture, cross-attention fusion, orthogonal contrastive loss, and custom Sharpness-Aware Minimization (SAM) optimization designed to completely eliminate numerical instability and NaN loss explosions.

### Overview & Architecture Upgrade
Unlike traditional single-stream image classifiers that frequently memorize training data or suffer from gradient explosions when scaled, FIV1.2 leverages a robust triple-stream hierarchical pipeline:

**Macro Stream (BioCLIP-2):** Leverages Imageomics' specialized Vision Transformer (hf-hub:imageomics/bioclip-2) to capture high-level semantic and taxonomic visual features.

**Micro/Texture Stream (ConvNeXt-Tiny):** Focuses on fine-grained surface details, scales, and bone textures with aggressive data augmentation and random erasing.

**Spatial Stream (ConvNeXt-Tiny):** Processes structural geometry and 3D mockups, backed by zero-mask safeguarding to prevent blank spatial inputs from polluting attention.

### Key Technical Features & Stability Patches

* **Hierarchical Taxonomic Safety Net:** Automatically queries the live Paleobiology Database (PBDB) API, climbing taxonomic ranks ($\text{Species} \to \text{Genus} \to \text{Family} \to \text{Superfamily} \to \text{Subfamily} \to \text{Infraorder}$) to provide graceful fallbacks when confidence falls below set thresholds ($85\%$ / $50\%$).
* **Cross-Attentive Fusion & Orthogonal Loss:** Combines embedding streams via multi-head self-attention while applying an orthogonal contrastive penalty ($\mathcal{L}_{\text{ortho}}$) to prevent representation collapse across streams.
* **Custom SAM Optimizer with Finite-State Guards:** Implements Sharpness-Aware Minimization with strict `torch.isfinite` gradient verification, dynamic norm clamping, and automatic weight rollbacks (`old_p`) if an adversarial perturbation goes unstable.
* **Logit Bounding (`safe_logits`):** Hard-clamps classification outputs between $[-30.0, 30.0]$ and sanitizes `NaN`/inf values prior to cross-entropy calculation under mixed-precision (`bfloat16`/`float16`) training.
