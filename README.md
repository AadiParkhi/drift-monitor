# Drift Monitor

Label-free drift monitoring for deployed computer-vision models, with root-cause attribution and accuracy-loss estimation.

Given a batch of about 100 images, the system answers three questions without any labels:

1. **Is this batch different** from clean data?
2. **What most likely caused it?** One of 15 image corruptions (blur, fog, noise, ...) or a change in the scene mix.
3. **How much accuracy will the deployed model lose?**

It returns one of three alerts:

| Level | Meaning |
|---|---|
| NONE | Nothing unusual |
| WATCH | Drift detected, expected loss under 5 accuracy points |
| ACT | Drift detected, expected loss of 5 points or more |

## How it works

- **Domains:** Places365 (48 scene classes) and BDD100K (11 weather x time-of-day conditions).
- **Deployed model:** DINOv2-small. **Canary sensor:** ConvNeXt-Tiny (an independent second model).
- **Detection:** per model, embeddings are standardized and reduced by PCA to 32 dimensions, then scored by a K-S test and an autoencoder reconstruction error (4 checks in total). Limits give about a 1% false-alarm budget and are calibrated for batch sizes 30, 50 and 100.
- **Cause:** a random forest over 15 corruptions plus a "mix shift" class (BDD100K).
- **Impact:** a regressor for corruptions, and a label-shift (BBSE-style) estimate for scene-mix changes.

## Results

| Measure | Places365 | BDD100K |
|---|---|---|
| Cause attribution on corruptions (held out) | 96.1% | 99.9% (before the mix-shift class was added) |
| Accuracy-loss estimate, R2 (held out) | 0.893 | 0.837 |

- False alarms on clean batches: about 0.5-2% at batch sizes 30, 50 and 100 (1-4% on unseen BDD100K data).
- Scene-mix impact error: 0.033-0.056 with the label-shift estimate, versus 0.051-0.124 for the corruption regressor alone.

## Repository layout

```
app/          Streamlit dashboard (app.py, core.py, ui.py, drift_pipeline.py)
  samples/    Sample batches and EXPECTED.csv (the alert each sample should give)
  models/     Put the model files here (not included, see below)
notebooks/    One notebook per project phase (run on Kaggle)
```

## Run the dashboard

1. Install the requirements (scikit-learn is pinned because the saved models were trained with that version):

   ```
   cd app
   pip install -r requirements.txt
   ```

2. Place these files in `app/models/`. They are too large for this repository:

   ```
   multitask_dino.pt
   multitask_convnext.pt
   detector_objects.joblib
   drift_artifacts.joblib
   dinov2_config/
   ```

   [Add a download link here once the files are uploaded.]

3. Start the app:

   ```
   streamlit run app.py
   ```

   Pick a domain, choose a sample batch (or upload about 100 images), and click **Analyze batch**.

## Notebooks

Run in this order on Kaggle (the paths in them point to `/kaggle/input/...`):

| Notebook | Purpose |
|---|---|
| 01a, 01a2 | Curate Places365 and BDD100K |
| 01b | Train the multitask models |
| 02 | Generate the corrupted images |
| 03a | Extract embeddings |
| 03b, 03c | Drift detection and the two-sensor design |
| 04 | Cause attribution and impact estimation |
| 05 | Alerting and limits |
| 06 | Package the dashboard |

## Limitations

- The corruptions are synthetic and made from small base image sets.
- A new camera or location may get a wrong cause, even when the alarm is right.
- The rain-heavy scene-mix estimate is the weakest case.
- Impact models were trained on 100-image batches, so other sizes are indicative.
- The scene-mix estimate exists only for BDD100K.

## Data

The datasets are not included. Get Places365 and BDD100K from their official sources and follow their licences.
