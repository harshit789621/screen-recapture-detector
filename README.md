# 🔍 Screen Recapture Detector

> **SalesCode AI — Computer Vision Take-Home Assignment**
> Detects whether an image is a genuine photograph or a photo-of-a-screen (recapture fraud).

---

## Demo

<!-- After recording your demo video, upload it to YouTube / Google Drive
     and replace the link below. For a local video, use:
     
     https://user-images.githubusercontent.com/YOUR_ID/YOUR_VIDEO.mp4
     
     GitHub supports MP4 embeds directly in markdown. -->

> 📹 **[Watch the demo video here](YOUR_DEMO_LINK)**  
> *(Single image and bulk upload walkthrough — Streamlit app)*

<!-- If you have an MP4 file, you can also drag-drop it into the
     GitHub README editor and it will embed automatically. -->

---

## Approach · Results · Cost

### How it works

The detector is built entirely on **classical computer-vision features** — no large neural network, no cloud API. The key physical insight is that a re-photographed screen is light emitted by a pixel grid, captured by a second camera. This double-pass leaves four measurable artifacts that real photos simply do not have:

| Feature family | What it catches | Dim |
|---|---|---|
| **FFT frequency analysis** | Screen pixel-grids create a bump + narrow moiré spikes in the frequency domain; real photos have smooth 1/f falloff | 18 |
| **LBP texture histogram** | The screen-door sub-pixel pattern shows up as an unusually regular local micro-texture | 18 |
| **Color statistics** | The second camera + backlit panel shifts white balance, saturation, and dynamic range | 12 |
| **Sharpness / edge density** | Autofocus hunting on a flat glossy surface softens fine detail vs a direct real-world shot | 2 |

A **50-feature vector** is extracted from each image and fed into an **RBF-kernel SVM** (`C=30, γ=0.01`) tuned via 5-fold grid-search. All preprocessing (center-crop to square → resize to 768 × 768) runs before feature extraction so image orientation and resolution cannot leak as confounding signals.

Training used **data augmentation on the training split only** (7 augmented copies per image: random rotation ±8°, brightness/contrast jitter, occasional Gaussian blur, JPEG re-compression at random quality) to give the model ~688 effective training samples from ~85 original photos, without any leakage into the test set.

### Accuracy

| Split | Images | Accuracy |
|---|---|---|
| 5-fold CV (training set) | 688 (augmented) | **91.0 %** |
| **Held-out test set** | **29 originals** | **93.1 %** |

```
Confusion matrix (0 = real, 1 = screen):
              Predicted
              Real   Screen
Actual Real  [ 13      0  ]   ← zero false accusations of real users
       Screen[  2     14  ]
```

The model never falsely flags a real photo as fraud (13/13 correct), while catching 87.5 % of actual recapture attempts. The two remaining misses are the hardest edge cases: a recapture where the screen fills only ~40 % of the frame (diluting the pixel-grid signal after preprocessing) and a printed-paper recapture (which lacks a regular pixel grid entirely — a structurally different artifact).

*Note: with only 29 test images each point is worth ~3.4 %, so the true accuracy has meaningful variance. A larger held-out set would pin it down more precisely.*

### Latency

| Scenario | Latency |
|---|---|
| Feature extraction + SVM inference (model already in memory) | **~400 ms / image, laptop CPU** |
| Cold-start CLI call (`python predict.py image.jpg`) | ~1.5 – 2 s (includes Python startup + `joblib.load`) |

In a real deployment the model is loaded once at server startup, so **~400 ms** is the number that matters.

### Cost per image

| Deployment | Cost per 1 000 images | Assumptions |
|---|---|---|
| **On-device** (user's phone, future port) | **~$0** | No server needed |
| Cloud CPU (AWS `t3.medium`, ~$0.04/hr) | **~$0.004 – 0.009** | 2 vCPUs, ~5 img/s parallelised; $0.04/hr ÷ 18 000 img/hr |
| Cloud CPU (`t3.large`, ~$0.08/hr) | **~$0.004** | 2 vCPUs, similar throughput |

SVM + classical CV runs entirely on CPU — no GPU required, no third-party API calls, no per-call billing.

### What I'd improve with more time

1. **Screen-region auto-detection before feature extraction.** The two persistent failures share a root cause: the actual screen occupies a small fraction of the frame, so the pixel-grid signal is diluted after the fixed-size resize. A lightweight object-detector (or even a simple edge-based rectangle finder) to crop to the screen region first would directly fix this.

2. **Dedicated printout / paper-texture feature.** The current FFT + LBP features are tuned for LCD/AMOLED pixel grids. Printed-paper recaptures have a halftone dot pattern — a different periodic frequency — that the current feature set is not specifically hunting. Adding a halftone-frequency detector would close this gap.

3. **Larger and more varied dataset.** ~115 original images is enough to demonstrate the approach but limits generalization to edge cases (extreme angles, very dark screens, unusual screen types). With 500+ images across a wider variety of devices and lighting the model would be more robust.

4. **Adaptive threshold.** Rather than a fixed 0.5 cutoff, the threshold should be calibrated on a separate validation set to balance false-positive rate (blocking real users) vs false-negative rate (missing fraud), depending on how much each mistake costs the business.

5. **On-device deployment.** The 50-feature SVM compiles to ~80 KB and runs in ~400 ms single-threaded on a laptop CPU. On a phone it would need Core ML (iOS) or TFLite (Android) wrapping, but the model is small enough that this is feasible.

---

## Project structure

```
FAKE VS REAL/
├── predict.py        ← merged detector: all features + inference in one file
├── train.py          ← training pipeline (split → augment → grid-search SVM)
├── run.py            ← Streamlit GUI (single image + bulk upload)
├── model.joblib      ← trained SVM model (~81 KB)
├── requirements.txt  ← pip dependencies
├── real/             ← 51 real photos used for training  [not in repo — large]
└── screen/           ← 64 screen/recapture photos        [not in repo — large]
```

---

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/screen-recapture-detector.git
cd screen-recapture-detector

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the CLI predictor
python predict.py path/to/image.jpg
# → prints a float 0.0 – 1.0  (0 = real, 1 = screen)

# 5. Launch the Streamlit GUI
streamlit run run.py
```

### To retrain on your own dataset

```bash
# Put your photos in  real/  and  screen/  folders, then:
python train.py --real real/ --screen screen/ --n_aug 7
# Saves a new model.joblib in the current folder
```

---

## Tech stack

`Python 3.11+` · `OpenCV` · `scikit-learn` · `scikit-image` · `NumPy` · `Streamlit` · `Pandas`

---

*Assignment submitted by **Harshit Pundir** · Bennett University, B.Tech CSE 2027*
