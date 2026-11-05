[Skip tocontent](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/33rd-place-silver-solution-no-external-data#site-content)

33rd Place (Silver) Solution – No External Data
===============================================

Series-level attention MIL + 3D RoPE + a training-only trap for site shortcuts

Solution Writeup · 32nd place · Oct 23, 2026

Thanks to RSNA, the annotators and Kaggle for this competition, and to everyone who shared notebooks and ideas.

This post covers only the main components of our strongest single model. Our final submission was a rank ensemble of models from this same family (Public 0.966 / Private 0.952); the ensemble itself is not described here.

Summary
-------

*   **No external data.** Training used only the competition data (images + reports). The only outside resources are public pretrained weights (OrthoFoundation-L, timm CoAtNet).
    
*   **Best single model:** Public 0.962 / Private 0.947 (OrthoFoundation-L), Public 0.961 / Private 0.948 (CoAtNet, same recipe).
    
*   Main components:
    
    1.  Two-level attention MIL over **all series and all slices** of a study, trained only with the 12 study-level labels.
        
    2.  Feature-map cells (~12 mm) as MIL instances, each with 3D patient coordinates, plus a 3D RoPE self-attention that links cells across series.
        
    3.  A **training-only "trap" branch** that absorbs site-specific reporting habits in the report-derived labels. This was our largest single gain.
        

Labels (briefly)
----------------

*   Training labels come from the reports. The 58 image-labelled studies were not used as training samples; we used them for validation and to calibrate the report labels.
    
*   An LLM extracts structured findings from each report (severity, "not mentioned" vs "explicitly negative", acute/chronic, …). The extracted levels (ours and those of [@pilkwang](https://www.kaggle.com/pilkwang)'s public LLM labels) are mapped to the positive rate among the 58 image-labelled studies (shrunk toward the overall rate) and averaged → soft targets.
    
*   When a report does not mention a finding, the target was almost constant (e.g. 75% of Synovitis targets). For those cells only, we spread the targets using earlier models of this family as teachers: y' = sigmoid(logit(y) + 0.5 \* z), with z the teacher logit standardized per finding.
    

Input
-----

*   All series and all slices; no series selection. Central 140 mm × 140 mm crop → 336 px (upsampled to 384 on GPU), per-series 2–98 percentile intensity scaling.
    
*   One window = 3 slices (center and ±3.5 mm) stacked as RGB.
    
*   Training: 4 random windows per series, 8 studies per batch. Inference: every window (~140 per study), no TTA.
    
*   From the DICOM geometry, every feature-map cell gets its 3D patient coordinate in mm, centered per study.
    

Model
-----

### Two-level attention MIL

*   **Level 1:** the bag is every cell of every window in one series. A small attention MLP (1024 → 256 → tanh → 12) gives one attention map per finding → 12 series vectors.
    
*   **Level 2:** per finding, attention over the series of the study, then a per-finding linear layer.
    
*   Studies with any number of series (3–14) and slices go in as they are. Nothing is dropped or squeezed into fixed slots.
    
*   Using cells instead of the pooled window feature lets attention pick a ~12 mm region, so small lesions (meniscal tears, fractures) are not averaged away with the whole slice.
    
*   The per-finding attention follows [@dreaddevelopment](https://www.kaggle.com/dreaddevelopment)'s public Raptor notebook.
    

### 3D RoPE across series

*   All series of a study share one patient coordinate system, so a sagittal cell and a coronal cell at the same 3D location can be linked directly.
    
*   Each head's 32 dims are split into x / y / z and rotated with wavelengths 4–256 mm, so the attention scores depend only on the relative 3D offset in mm.
    
*   The output is zero-initialized and added back to the 4 underlying cells, so training starts exactly from the plain cell model.
    
*   To be transparent: when we added this block on top of cell attention (older CoAtNet recipe), the LB moved by +0.002 public / +0.001 private, which is within our run-to-run noise. We kept it in the final recipe.
    

### Backbone

*   **OrthoFoundation-L** (ViT-L/16; DINOv3-L further trained self-supervised on knee images). 24×24 tokens → 2×2 average → 12×12 cells. Trained parts: last 8 of 24 blocks, final norm and patch embedding (lr 3e-5; head 1e-3).
    
*   The patch embedding was initialized to see only the center slice (RGB weights summed into the center channel, neighbor channels set to zero). After training, the neighbor-slice weights were only ~2% of the center weights. So in practice the backbone reads single slices, and context across slices comes from the MIL and RoPE parts.
    
*   Tip: keep DINOv3's own 2D RoPE enabled and use the RoPE periods stored in the checkpoint. Then our timm features match the official DINOv3 code exactly. With model.rope = None (as in some public notebooks), the features deviate strongly (max abs diff 2.88). We did not measure the effect on the score.
    
*   On T4 with fp16, rows that overflow are recomputed in fp32.
    
*   **CoAtNet** (coatnet\_rmlp\_2\_rw\_384.sw\_in12k\_ft\_in1k) with the same recipe scored on par (Public 0.961 / Private 0.948 vs 0.962 / 0.947), with about 1/4 of the backbone FLOPs.
    

Shortcut learning: site-specific reporting habits
-------------------------------------------------

**What we saw**

*   Training labels come from reports, each written by one radiologist at each site. Test labels come from MSK radiologists reading the images with one strict rule.
    
*   Positive rates in the report labels differ a lot by report language (e.g. PF OA 0.14–0.43).
    
*   Acquisition parameters alone (field strength, slice thickness / spacing, pixel size, …) predict the report language with 98% accuracy (5-fold CV; majority class 39%).
    
*   An earlier model received these parameters through FiLM and a term in the series attention. Turning those parts off **at inference only** (same weights) moved the LB from Public 0.947 / Private 0.929 to Public 0.959 / Private 0.944. The model had been using "which site is this?" to reproduce each site's reporting habit.
    

**The trap (training only)**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   training:   logit = image_logit + a[site, finding] + mean over series of g(acquisition params)[finding]  inference:  logit = image_logit   `

*   **Site** = pseudo-site: report language × scanner (vendor / model / field strength). Groups with fewer than 30 studies are merged → 44 pseudo-sites.
    
*   **$a$** is treated like a random effect: a Gaussian prior N(0, 0.5²) as a penalty, initialized at the shrunk (empirical-Bayes) site log-odds of the labels. So from the first step, the image path learns mostly from differences within each site.
    
*   **$g$** is a small MLP on the acquisition parameters (→ 32 → 12, zero-init output), so that "how it was scanned" can be absorbed by the trap and not by the image path.
    
*   The idea is close to bias-only branches (product-of-experts debiasing) and to TR (arXiv 2608.00732).
    
*   Limitation: the image path can still recognize the site from image appearance. The trap only gives the shortcut a cheaper place to go.
    

**Single-model results** (OrthoFoundation-L, same schedule, one change per row)

**ModelPublicPrivate**No acquisition parameters, no trap0.9490.932+ site term $a$0.9550.940+ acquisition branch $g$, empirical-Bayes init0.9600.946+ spread "not mentioned" targets (see Labels)0.9620.947Same, CoAtNet backbone0.9610.948

In all 4 pairs where we removed the acquisition-parameter parts at inference, the gain on private (+0.009 to +0.015) was slightly larger than on public (+0.007 to +0.012).

Training / inference
--------------------

*   ~4,350 studies with report-derived labels, weighted BCE on soft targets, 16 epochs (one-cycle), SWA over epochs 14–16, A100 bf16.
    
*   Inference on Kaggle T4 (fp16), all windows, no TTA: about 10–16 s per study per model.
    

Author[hirokenn](https://www.kaggle.com/hirokenn)

hirokenn

Share

Citation
--------

hirokenn. 33rd Place (Silver) Solution – No External Data. https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/33rd-place-silver-solution-no-external-data. 2026. Kaggle

3

0 Comments
----------

This comment will be made public once posted.

Post Comment