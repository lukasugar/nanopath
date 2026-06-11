# Proposed experiments

Context: Nanopath is a small, single-GPU pathology SSL harness. The public score to beat is `mean_probe_score`, and a real gain should be at least +0.006. The live Labless log currently shows `lr-and-curation` as main at 0.6357, validated `I-JEPA contig patch` at 0.6444, and pending JEPA mask variants around 0.6470-0.6501.

Prior results suggest three patterns:
- DINOv2 + KDE is a strong baseline, but the JEPA patch-regression swap is the largest recent jump.
- JEPA improves linear/kNN/few-shot/robustness but has hurt segmentation unless balanced carefully.
- Mask scale 0.05-0.25 is already tested; 0.10 is best so far, but the local optimum is not resolved.

Literature anchors:
- I-JEPA argues for semantic target blocks and informative context, making mask geometry a first-class knob: https://arxiv.org/abs/2301.08243
- DINOv2 combines curated data, self-distillation, and robust dense features, matching Nanopath's DINO/KDE base: https://arxiv.org/abs/2304.07193
- iBOT's patch self-distillation is strong for dense tasks, so replacing it with JEPA should be watched for segmentation regressions: https://arxiv.org/abs/2111.07832
- DINOv3's Gram-anchoring result points at dense-feature preservation as a useful pressure, even though the current live `jepa-gram-anchor` attempt underperformed: https://arxiv.org/abs/2508.10104
- Pathology SSL papers repeatedly emphasize stain/domain robustness, supporting careful rather than maximal stain jitter: https://arxiv.org/abs/2208.04017 and https://arxiv.org/abs/2211.07590

## Candidate pool

1. `jepa-mask075`: interpolate between the strong 0.05 and 0.10 block scales with `jepa_block_scale: 0.075`.
2. `jepa-mask15`: interpolate between 0.10 and 0.25 with `jepa_block_scale: 0.15`, checking whether the optimum is broader than the current best.
3. `jepa-tissue03`: combine the validated JEPA recipe with modest tissue rejection (`tissue_thresh: 0.3`) to try to recover segmentation without returning to the harsher main curation.
4. `jepa-local126`: increase local crop size to 126 so the eight local views carry more morphology and spatial context.
5. `jepa-half-loss`: add `jepa_loss_weight: 0.5` so DINO/KDE carries more of the representation while JEPA remains a dense-context regularizer.
6. `jepa-beta025`: add `jepa_beta: 0.25` for smooth-L1 patch regression, making ambiguous masked targets less purely quadratic.
7. `jepa-blocks6`: raise `jepa_blocks` from 4 to 6 at scale 0.10 for more spatially distributed targets.
8. `jepa-crop35`: relax global/local crop floors to `[0.35, 1.0]` and `[0.07, 0.35]`, halfway back toward the DINO recipe.

## Top 5 to implement

1. `codex/jepa-tissue03` - highest upside because it directly addresses the JEPA leader's weakest metric, segmentation.
2. `codex/jepa-mask075` - cheap config-only refinement around the live best mask scale.
3. `codex/jepa-mask15` - cheap config-only refinement on the other side of the live best mask scale.
4. `codex/jepa-local126` - tests whether more local morphology helps few-shot/segmentation without changing objective code.
5. `codex/jepa-half-loss` - minimal code/config change to test whether JEPA is currently too dominant.

Run command for each branch:

```bash
RUN_DIR=/data/$USER/nanopath/main/<run-name>
./submit/train_1gpu.sbatch configs/main.yaml output_dir=$RUN_DIR
```
