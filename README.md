# FIV1.2

### Work in Progress

Building upon the original single-stream ConvNeXt baseline, FIV1.2 introduces an advanced triple-stream architecture, cross-attention fusion, orthogonal contrastive loss, and custom Sharpness-Aware Minimization (SAM) optimization designed to completely eliminate numerical instability and NaN loss explosions.

## Architecture

### 1. System Architecture & Stream Pipeline

FIV1.2 utilizes an advanced triple-stream multi-modal architecture designed to process macro photographs, micro-textures, and 3D spatial geometry in parallel before fusing them into a unified representation.

* **Stream 1 (Macro Morphology):** Powered by BioCLIP-2 Vision Transformer (`hf-hub:imageomics/bioclip-2`), extracting global semantic visual representations, projected down to 256 dimensions via LayerNorm and GELU activations ($z_1$).
* **Stream 2 (Micro-Texture Analysis):** Powered by ConvNeXt-Tiny optimized for fine surface features, grain patterns, and matrix textures, projected to 256 dimensions ($z_2$).
* **Stream 3 (3D Spatial Geometry):** Powered by a second ConvNeXt-Tiny backbone configured for depth maps or normal scans, projected to 256 dimensions ($z_3$). Includes an automated zero-tensor guard (`spatialIN.abs().sum() == 0`) which forces $z_3 = 0$ if 3D data is missing, preventing backward-compatibility failures on standard 2D field archives.

### Cross-Attentive Fusion Neck
Integrates streams using multi-head attention followed by a gating mechanism:

$$\text{gate} = \sigma\left(\text{Linear}([z_1; z_2; z_3])\right)$$

$$\text{fused} = \left( \text{LayerNorm}(\text{Attention}([z_1, z_2, z_3]) + [z_1, z_2, z_3]) \times \text{gate} \right).\text{reshape}(\text{Batch}, -1)$$

---

### 2. Multi-Task Hierarchical (MTL) Heads & Loss Calculations

The model features six concurrent classification heads (Species + 5 PBDB taxonomic ranks) using Multi-Sample Dropout (`MSDhead`, `numsamples1=5`, $p=0.2$) to stabilize training on sparse data.

* **Total Network Loss ($\mathcal{L}_{\text{total}}$):** Combines label-smoothed cross-entropy across all taxonomic heads with hierarchical rank weighting and an orthogonal regularization penalty:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{species}} + \sum_{r \in \text{Ranks}} w_r \mathcal{L}_{\text{rank}_r} + \mathcal{L}_{\text{ortho}}$$

* **Species Classification Loss ($\mathcal{L}_{\text{species}}$):** Standard Cross-Entropy with label smoothing ($\epsilon = 0.1$):

$$\mathcal{L}_{\text{species}} = -\sum_{i} y_i \log \left((1 - \epsilon)\hat{y}_i + \frac{\epsilon}{C}\right)$$

* **Hierarchical Rank Weights ($w_r$):**
  * Genus: $w_{\text{genus}} = 0.5$
  * Family: $w_{\text{family}} = 0.25$
  * Superfamily / Subfamily / Infraorder: $w_r = 0.15$

* **Orthogonal Stream Contrastive Loss ($\mathcal{L}_{\text{ortho}}$):** Prevents representation collapse by penalizing cosine similarity between normalized stream projections ($\hat{z}_1, \hat{z}_2, \hat{z}_3$):

$$\mathcal{L}_{\text{ortho}} = \frac{\alpha}{3} \left( \left|\langle \hat{z}_1, \hat{z}_2 \rangle\right| + \left|\langle \hat{z}_1, \hat{z}_3 \rangle\right| + \left|\langle \hat{z}_2, \hat{z}_3 \rangle\right| \right)$$

*(where $\alpha = 0.1$, and $\hat{z} = \frac{z}{\|z\|_2 + 10^{-8}}$)*

---

### 3. Optimization Math & Mixed-Precision Protocol (`FIV1.2` Stability Upgrades)

To train safely on small datasets without VRAM leaks or `NaN` gradient explosions, FIV1.2 implements a customized Sharpness-Aware Minimization (SAM) optimizer wrapped around AdamW, executed under PyTorch mixed precision (`torch.amp.autocast`) with robust finite-state guards.

### SAM Two-Step Perturbation Calculus
1. **First Forward-Backward Pass:** Computes standard gradients $\nabla_W \mathcal{L}$ under autocast scaling:
   $$\mathcal{L}_{\text{scaled}} = \text{scaler.scale}(\mathcal{L})$$
2. **Finite-State Gradient Filtering (`_grad_norm`):** Validates that all gradients are finite (`torch.isfinite`) before calculating global L2 norms, preventing `NaN` propagation or division-by-zero.
3. **Gradient Unscaling Safeguard:** Explicitly strips the dynamic scaling factor before calculating geometric norms:

   `scaler.unscale_(optimizer.base_optimizer)`

4. **Adversarial Perturbation Vector ($\epsilon$):** Evaluates true geometric gradient norms to shift weights into an adversarial neighborhood:

   $$\hat{\epsilon} = \rho \frac{\nabla_W \mathcal{L}}{\|\nabla_W \mathcal{L}\|_2 + 10^{-12}}$$

   $W_{\text{perturbed}} = W + \hat{\epsilon} \quad \text{(with old p safely cloned)}$

5. **Second Forward-Backward Pass:** Computes loss at the perturbed weights:

   $$\mathcal{L}_p = \mathcal{L}(W + \hat{\epsilon})$$

   $$\text{scaler.scale}(\mathcal{L}_p).\text{backward}()$$

6. **Defensive Weight Rollback Check:** If `loss_perturbed` is non-finite (`not torch.isfinite`), training automatically aborts the update, restores weights from `old_p`, clears gradients, and skips the batch.
7. **Parameter Update:** Applies the base optimizer step via the GradScaler:

   $$W \leftarrow W - \eta \nabla_{W} \mathcal{L}_p$$

* **Safe Logit Bounding (`safe_logits`):** All hierarchical heads pass through a safety wrapper prior to loss evaluation, clamping outputs between $[-30.0, 30.0]$ and sanitizing `NaN`/infs:

$$\text{logits safe} = \text{clamp}(\text{nan to num}(\text{logits}), -30.0, 30.0)$$

---

### 4. OOD Handling & PBDB Phylogenetic Fallback Logic

To handle out-of-distribution items (e.g., modern bones, rocks, non-fossil objects) and data starvation, inference applies strict safety floors and API-driven tree traversal:

* **Inference Confidence Floor ($15\% / 0.15$):** Any prediction landing below a 15% softmax score is intercepted immediately and categorized as Unidentified Dinosauria (clade).
* **Taxonomic Tree-Climbing Fallback:**
  * If confidence sits in the medium tier ($50\% \le c < 85\%$), the engine climbs the Paleobiology Database (PBDB) parent tree to extract the accepted genus or subfamily.
  * If confidence sits in the low tier ($15\% \le c < 50\%$), the engine bypasses lower nodes entirely ("Genus Trap" avoidance) and climbs to broad clades (family, superfamily, or infraorder).
* **Head-Specific Probability Extraction:** Instead of recycling the weak species score, probabilities for fallback categories are queried directly from the model's corresponding multi-task taxonomic head via explicit index mapping:
