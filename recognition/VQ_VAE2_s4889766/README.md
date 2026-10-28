# VQ-VAE2 on HipMRI Dataset

**Author:** Johnathan Ngo\
**Student ID:** s4889766\
**Difficulty:** Hard

## VQ-VAE2 Architecture
[placeholder]


## Dependencies
[placeholder]

## Dataset and Preprocessing

## Training Procedure
[placeholder]

### Loss Function
[placeholder]

## Results
Every setting below draws 512 samples from the same seed. Two measures are used throughout:

- **FID** against the 11,460 training slices, where lower is more realistic. Real slices from the three unseen test patients score 62.6, roughly where a generator of genuinely new patients would sit.
- **The nearest-neighbour audit.** For each sample, this is the SSIM to its most similar training slice. The reference is the same measure for the test patients' slices, passed through the same decoder so that its blur affects both sides equally. A sample is **flagged** when it is closer to a training slice than any test slice gets (the reference maximum, 0.435–0.439 depending on the model). A flagged sample is closer to the training data than a new patient's scan is, which is the signature of memorisation.

### Shared sampling temperature
Each model was first sampled with one temperature, $T$, shared by all of its priors. Three models are compared: the single-level baseline, VQ-VAE-2, and VQ-VAE-2 trained with an extra SSIM term in its reconstruction loss (weight 0.01).

| Model | $T = 0.8$ | $T = 0.9$ | $T = 1.0$ |
|---|---|---|---|
| Baseline (1 level) | 150.1 | **138.1** | 161.4 |
| VQ-VAE-2 | **130.0** | 147.7 | 186.1 |
| VQ-VAE-2 + SSIM loss | **120.1** | 129.5 | 159.0 |

*FID against the training slices (lower is better), with each model's best in bold.*

<img src="assets/results/fid_shared.png" width="70%">

- **At its best temperature, each two-level model beats the baseline.** VQ-VAE-2 reaches 130.0 and the SSIM-loss model 120.1, against 138.1 for the baseline.
- **The two-level model is more sensitive to temperature.** It is best at 0.8 and worst at 1.0, where it falls behind the baseline (186.1 against 161.4). A likely reason is that both of its priors sample at $T$, so a hot setting gives a less coherent layout and then noisier texture on top of it.
- **The SSIM loss lowers FID at every temperature,** by 9.9 at 0.8, 18.2 at 0.9 and 27.1 at 1.0.
- **Every model remains far from the floor.** The best score (120.1) is nearly twice the 62.6 of real test slices. The decoder accounts for only part of that gap: the plain VQ-VAE-2's reconstructions of the test slices score 78.2. Most of it therefore comes from the priors.

<img src="assets/results/samples_shared_baseline.png" width="100%">

*Uncurated samples from the baseline: the first six of 512 at each temperature.*

<img src="assets/results/samples_shared_vqvae2.png" width="100%">

*Uncurated samples from VQ-VAE-2, laid out the same way.*

The grids match the scores. At $T = 0.8$, half of the VQ-VAE-2 samples show a bright bladder inside a recognisable pelvis, with the femoral heads on either side. At 0.9 and 1.0, most become a tangle of tissue texture with no clear organs. The baseline's samples are smoother and lower in contrast, several frames are foggy, and some of its bladders at $T = 1.0$ are misshapen.

### Memorisation
At a lower temperature, the priors favour their most likely codes, and those are the codes that rebuild training slices. Realism and memorisation can therefore rise together, so every setting was audited.

**Where the samples sit.**

<img src="assets/results/tsne_baseline.png" width="49%"> <img src="assets/results/tsne_vqvae2.png" width="49%">

*t-SNE of Inception features for the training slices (grey), the test slices (orange) and the samples at each temperature, for the baseline (left) and VQ-VAE-2 (right).*

- **No sample lands on a training slice.** In both models, the samples form their own cloud beside the real slices. A memorising model would instead place samples on top of individual training points.
- **The gap between the clouds is the realism gap that FID measures.** Inception features still separate generated slices from real ones. The test slices, in contrast, sit along the edge of the training cloud in short streaks, likely runs of neighbouring slices from one patient.
- **There is no sign of mode collapse.** The three temperatures overlap within the sample cloud rather than splitting into tight clusters. VQ-VAE-2's $T = 1.0$ samples drift to the far side, away from the real data, in line with their FID of 186.1.

**How close the samples get.**

<img src="assets/results/memorisation_shared.png" width="100%">

*Left: the median SSIM between a sample and its nearest training slice; the grey band is the same median for the test patients. Right: the number of samples closer to a training slice than any test slice gets.*

- **VQ-VAE-2 at $T = 0.8$ is closer to the training set than a new patient is.** Its typical sample has a median nearest-train SSIM of 0.360, against 0.324 for the typical test slice. 38 of its 512 samples (7%) are flagged. Both measures fall as $T$ rises, to 11 flagged at 0.9 and 1 at 1.0.
- **The SSIM loss halves the flags at $T = 0.8$** (18), while also improving FID.
- **The baseline stays at or below the reference at every temperature,** with 1–3 samples flagged. Its weaker prior produces samples that are further from every real slice, not only from the training slices.

Under a shared temperature, the setting where the two-level model is most realistic is therefore also the one where it is closest to its training data.

**What a flagged sample looks like.**

<img src="assets/results/nn_pairs_t0.8_vqvae2.png" width="100%">

<img src="assets/results/nn_pairs_t0.8_baseline.png" width="100%">

*The six samples closest to the training set at $T = 0.8$ (top row of each figure), above their nearest training slice (bottom row). The first figure is VQ-VAE-2 and the second is the baseline.*

- **None is a copy.** The flagged samples share their neighbour's layout, such as the size and position of the bladder and the pelvic outline, but not its detail.
  - Their SSIM is 0.47–0.49. The decoder's reconstruction of a slice scores 0.94 against the original, which is roughly what a copy would score.
  - At the code level, no sample matches a training slice exactly. The closest shares 18.9% of its top codes and 29.1% of its bottom codes with a training slice, against 11.1% and 20.1% for the test slices.
- **The closest samples come from few patients.** VQ-VAE-2's six trace back to three training patients (018, 026 and 027). Four of the baseline's six trace back to slices 29–32 of patient 027.
- **The baseline has favourite patients too, even though few of its samples are flagged.** At the code level it is the closer model: its closest sample shares 33.8% of its codes with a training slice, against 29.1% for VQ-VAE-2's bottom level and 20.2% for the test slices.

### Separate temperatures
A shared temperature ties together two different choices. The top prior decides the layout and the bottom prior the texture. Cooling both to sharpen the texture also pulls the layouts toward the training set. Cooling further makes things worse:

- **VQ-VAE-2 at $T = 0.7$:** FID barely moves (131.3), while the flagged samples double to 75.
- **VQ-VAE-2 at $T = 0.6$:** FID rises to 149.9, with 105 samples flagged.
- **Baseline at $T = 0.6$:** the model collapses (FID 268.3) into off-centre, mostly blank frames.

Giving each prior its own temperature separates the two choices. The top prior can stay warm for varied layouts, while the bottom prior runs cold for clean texture. `predict.py sample --bottom-temperatures` samples every pairing of a top and a bottom temperature. Because every setting uses the same seed, settings that share $T_\text{top}$ also share their top code maps, so comparing them changes only the texture.

The first sweep used the SSIM-loss model, with $T_\text{top}$ and $T_\text{bottom}$ each in $\{0.7, 0.8, 0.9\}$:

| $T_\text{top}$ \ $T_\text{bottom}$ | 0.7 | 0.8 | 0.9 |
|---|---|---|---|
| 0.7 | 128.9 / 23 | 137.2 / 29 | 155.9 / 21 |
| 0.8 | 107.2 / 28 | 120.1 / 18 | 137.5 / 10 |
| 0.9 | **94.4 / 12** | 106.2 / 12 | 129.5 / 4 |

*Each cell gives the FID, then the number of flagged samples (of 512). The diagonal is the shared temperature.*

- **$T_\text{bottom}$ drives realism.** FID falls as $T_\text{bottom}$ falls, in every row.
- **$T_\text{top}$ drives memorisation.** Flags tend to fall as $T_\text{top}$ rises: at $T_\text{bottom} = 0.9$, they go from 21 to 10 to 4. A warm top prior keeps layouts away from the training set.
- **The best setting beats the best shared temperature on both measures.** At 0.9/0.7, FID is 94.4 with 12 flagged, against 120.1 with 18 at $T = 0.8$.
- **Both trends were still improving at the edge of the grid,** at the coldest bottom and the warmest top temperature, so the grid was extended in those directions.

<img src="assets/results/samples_separate.png" width="100%">

*Uncurated samples for each setting of the first sweep: the first six of 512. Rows with the same $T_\text{top}$ share their top codes, so each column keeps its overall layout across those rows and only the texture changes.*

### Extending the separate temperatures
The second sweep took $T_\text{top}$ up to 1.0 and 1.1 and $T_\text{bottom}$ down to 0.6 and 0.5. It reused the samples already drawn at 0.9/0.7. The full grid is below, in the same format as before.

| $T_\text{top}$ \ $T_\text{bottom}$ | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| 0.7 | | | 128.9 / 23 | 137.2 / 29 | 155.9 / 21 | |
| 0.8 | | | 107.2 / 28 | 120.1 / 18 | 137.5 / 10 | |
| 0.9 | 82.4 / 10 | 84.5 / 20 | 94.4 / 12 | 106.2 / 12 | 129.5 / 4 | |
| 1.0 | 77.3 / 11 | 81.7 / 11 | 93.3 / 4 | | | 159.0 / 4 |
| 1.1 | **77.9 / 0** | 85.3 / 2 | 96.4 / 6 | | | |

<img src="assets/results/tradeoff.png" width="85%">

*FID against flagged samples for every setting. Lines join each model's shared temperatures in order; the squares are the SSIM-loss model's separate settings. The lower left is better on both counts.*

- **Shared temperatures soon stop paying off.** As $T$ falls, each shared-temperature curve first moves left, then turns. The two-level models turn upward, gaining flagged samples without further realism. The baseline turns back to the right as its samples degrade.
- **The separate settings sit below and to the left of every curve,** the baseline's included. The best of them reach FIDs of 77–82, which no shared temperature comes near for any model.

<img src="assets/results/samples_extended.png" width="100%">

*Uncurated samples from the second sweep: the first six of 512 at each setting.*

Across the full grid, the two temperatures do different jobs:

- **$T_\text{bottom}$ sets realism.** In every row, FID falls as $T_\text{bottom}$ falls. At $T_\text{top} = 0.9$, it drops from 129.5 at $T_\text{bottom} = 0.9$ to 82.4 at 0.5.
- **$T_\text{top}$ sets novelty.** At a fixed $T_\text{bottom}$, the median nearest-train SSIM falls as $T_\text{top}$ rises. At $T_\text{bottom} = 0.7$, it goes from 0.360 at $T_\text{top} = 0.7$ to 0.302 at 1.1. Hotter top sampling produces layouts further from every training slice.
- **$T_\text{top}$ also matters for FID, up to a point.** At $T_\text{bottom} = 0.7$, raising $T_\text{top}$ from 0.7 to 0.9 lowers FID from 128.9 to 94.4, probably because colder top sampling repeats a narrower set of layouts. Above 0.9, FID is flat while novelty keeps improving.

**Recommendation: $T_\text{top} = 1.1$, $T_\text{bottom} = 0.5$.**

- **Realism:** its FID of 77.9 ties the lowest in the grid (77.3 at 1.0/0.5). One draw of 512 samples cannot resolve the 0.6 gap.
- **Memorisation:** no sample is closer to a training slice than the closest unseen test patient, against 11 at 1.0/0.5.
- **Novelty:** its median nearest-train SSIM (0.310) is below that of the test patients' reconstructions (0.321). The typical sample is therefore no closer to the training data than a new patient is.
- **Against the best shared temperature** ($T = 0.8$: FID 120.1, 18 flagged), FID falls by 42 and no samples are flagged.
- **Against the baseline at its best** ($T = 0.9$: FID 138.1, 2 flagged), FID falls by 60 and no samples are flagged.

<img src="assets/results/nn_pairs_t1.1_0.5.png" width="100%">

*The six samples closest to the training set at 1.1/0.5, above their nearest training slice.*

Even the closest samples at this setting stop at the reference maximum (SSIM 0.425–0.435). Like those at $T = 0.8$, they share only their neighbour's layout. They still trace back to the patients who dominated the shared-temperature audit, 026 and 027, which a patient-level audit should check.

**Caveats:**

- **Edge of the grid.** The optimum lies on the grid's edge: $T_\text{bottom} = 0.5$ is the coldest setting tried and $T_\text{top} = 1.1$ the hottest. Better settings may therefore lie outside the grid. Any search beyond it should be judged on the audit as well as FID, because FID is measured against the training slices and would also reward copies.
- **Selection on the same samples.** The setting was chosen as the best of 18 on the same samples it is reported with, so its FID is optimistic. A fresh draw with a new seed should confirm it before it is quoted as final.
- **Code agreement.** The code-level audit does not fully clear this setting.
  - No sample shares more than 17.8% of its top codes, or 24.1% of its bottom codes, with any training slice. That is far from a copy.
  - Both figures are still above the test patients' 10.2% and 16.2%, so at the code level the samples sit closer to the training data than new patients do.

## AI Usage Disclosure
[placeholder]


## Feasibility Review
### User Need, Scope & Acceptance Criteria
The users are researchers who need shareable synthetic HipMRI slices (256×128) without releasing real scans, which would otherwise violate patient confidentiality. The central dilemma is training a model that produces 'realistic' images, potentially for downstream tasks, while also ensuring that the model does not memorise training images.

The acceptance criteria are:

1. **Fidelity:** Test reconstruction is recorded at SSIM > 0.6.
2. **Realism:** FID recorded below the baseline model's FID.
3. **No memorisation:** No latent samples have a high SSIM with a training slice, verified via some nearest-neighbour search.
4. **Diversity:** No mode collapse in uncurated grids or t-SNE, and latent interpolations show no 'jumps'.

### Model Choice
VQ-VAE-2 (Razavi et al., 2019) trains in two stages. Stage 1 encodes each slice into 64×32 and 32×16 latents, replaces each latent vector $z$ with its nearest of $K = 512$ codes $e \in \mathbb{R}^{64}$, and decodes with loss

$$\mathcal{L} = \lVert x - \hat{x} \rVert_2^2 + \beta \lVert z - \mathrm{sg}[e] \rVert_2^2, \qquad \beta = 0.25,$$

where sg is the stop-gradient. The codes in the codebook are EMA-updated with decay 0.99 with unchosen codes resetting, i.e., moving to a randomly chosen encoder output (van den Oord et al., 2017).

Stage 1 leaves every image as a small grid of labels, where each label describes what one little patch of the image looks like. Instead of drawing 32,768 pixels, we only need to create two small grids of labels: 64×32 for the global detail and 32×16 for the finer detail. **Stage 2** learns how those labels are usually arranged, and walks the grid one label at a time. While doing so, the next label is predicted from everything it has already seen. With teacher-forced cross-entropy, the model corrects any wrong guesses as it pertains to the label prediction. The **top** prior draws the global grid first, which essentially produces the scans' outline and decides where the organs go. The **bottom** prior then fills in the fine grid with the top model as reference, so the detail matches the layout. 

To generate a new image, the top model picks labels one by one, with the bottom model following. The decoder then turns the two label grids back into an image. The temperature $T$ of sampling essentially determines how varied our generations can be, e.g. low values are more deterministic and repetitive, while high values are more varied / novel but also more error-prone since its so varied.

**Why VQ-VAE2 over a standard VQ-VAE:** 

An MRI slice has both a global layout (the outline of the body and the organs) and local texture in the tissue / bones. A single-level VQ-VAE's prior both places the organs while predicting 2,048 codes one at a time. This means that any early prediction errors can ultimately affect later predictions. With VQ-VAE-2, we delegate these steps to seperate priors such that we fix the layout in 512 steps with attention over the whole slice. Then, the bottom codes add detail on top of it. Therefore, I will consider the 1-level VQ-VAE to be the baseline model for the experiment.

### Preliminary Evidence of Feasibility
<img src="assets/feasibility_recon.png" width="80%">

- **Data audit:** 12,660 slices from 38 patients, split by patient (train 11,460 / 31 patients, validate 660 / 4 patients, test 540 / 3 patients).
- **Stage-1 test** (validation): after 6 minutes on an A100, SSIM 0.932 $\pm$ 0.011 and PSNR 31.6 dB.
- **Priors:** ~52 M parameter PixelCNN used, which samples at around 3s per image.

### Risks, Budget & Fallback
| Risk | Warning sign | Mitigation |
|---|---|---|
| Memorisation | Validation NLL rises | Use early stopping to prevent overfitting, and the nearest-neighbour audit |
| Unused Codes | Perplexity of codebook falls | Use EMA updates, and force some random restart of the codebook. |

**Budget:** 1x A100-40GB GPU for training and sampling. 

**Next steps:** 

1. Train the PixelCNN priors for sampling generated images in stage 2.
2. sweep the sampling temperature $T$ and measure FID.
3. Experiment with using some weighted loss that targets SSIM.
