# SR-SAR — Super Resolution for Synthetic Aperture Radar

**Bilkent University EEE Senior Design Project (Group C5), 2025–2026**
**Industry partner: Meteksan Savunma Sanayi**

🔗 [Official project page](https://ee.bilkent.edu.tr/fuar/2026/group_c5/project_page_c5.html) · 📄 [Project booklet (PDF)](https://ee.bilkent.edu.tr/fuar/2026/group_c5/c5_booklet.pdf)

---

## The problem

MILSAR synthetic-aperture radar produces imagery whose resolution is bounded by the radar hardware. Improving it physically means new hardware — expensive, slow, and often impossible for systems already in the field.

The harder constraint is what makes SAR different from ordinary image super-resolution:

- **Speckle noise and real structure look alike.** SAR speckle is a coherent-imaging artefact with statistics that resemble genuine scattering features. A denoiser that removes speckle also removes real targets.
- **Hallucination is unacceptable.** A model that invents plausible-looking detail is worse than no model at all in a defence context. The output has to be *interpretable*, not merely sharp.
- **The right trade-off depends on the task.** An analyst looking for edges of a structure wants different behaviour from one assessing terrain texture.

## The approach

A software-only pipeline producing **2× and 4×** super-resolved SAR output, with no modification to the radar.

### Architecture

Built on **Swin2-MoSE**, a Transformer-based super-resolution architecture, implemented in PyTorch.

### Dual-model strategy

Rather than forcing one network to balance competing objectives, two models are trained with different priorities:

| Model | Optimised for |
|---|---|
| **A** | Edge and structural preservation |
| **B** | Speckle-noise suppression |

### α-controlled blending

Outputs are combined through a weighted blending mechanism governed by a parameter **α**:

```
output = α · edge_preserving + (1 − α) · speckle_suppressed
```

This makes the fidelity/smoothness trade-off an **explicit, operator-controllable dial** rather than something frozen into the weights at training time. The analyst decides, per image and per task, where on that spectrum they want to sit.

### Pipeline

- **Patch-based preprocessing** — large-format SAR scenes are tiled for training and inference, then reassembled.
- **Training, validation and benchmarking** workflow for comparing model variants quantitatively.
- **GUI** for visualisation, side-by-side comparison and export — so radar engineers with no machine-learning background can run the system unassisted.

---

## Team

Ömer Faruk Akbay · Atahan Karabey · Arda Karaman · Yunus Emre Koçak · Mohammadhassan Mahmodi · Mehmet Paşa

**Academic supervisor:** Dr. Vakur B. Ertürk
**Teaching assistant:** Zeynep Ortahüner
**Industry mentor:** Aksay Fatih Öncel (Meteksan Savunma)

---

## Repository contents

<!-- TODO: fill in once code is uploaded -->

| Path | Contents |
|---|---|
| `src/` | *(to add)* Model definitions, training and inference scripts |
| `gui/` | *(to add)* Visualisation and export interface |
| `notebooks/` | *(to add)* Experiments and evaluation |
| `docs/` | *(to add)* Report and figures |

## Results

<!-- TODO: add before/after image pairs and quantitative metrics (PSNR / SSIM or the
     domain-appropriate SAR metrics used in the report). Reviewers look for this first. -->

*To be added — see the [project booklet](https://ee.bilkent.edu.tr/fuar/2026/group_c5/c5_booklet.pdf) in the meantime.*

## Notes

Work carried out under an industry collaboration; any code or data published here is limited to what the partner has cleared for release.
