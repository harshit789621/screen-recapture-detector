# Screen Recapture Detector

## Demo

[https://github.com/user-attachments/assets/REACPTURE_DETECTOR.mp4](https://github.com/user-attachments/assets/ad5fbd2d-f520-4de6-b50f-c0cccb0b3dbf)

---

## What it does

I built a detector that looks at an image and decides whether it is a **real photograph** or a **photo of a screen** (someone re-photographing a phone or laptop instead of the real thing). To do this, I extract four types of features from each image. The **FFT analysis** looks at the frequency pattern of the image — a screen always has a repeating pixel grid that shows up as unusual spikes in the frequency domain, while real photos have a smooth natural pattern. The **LBP texture** captures the fine repetitive micro-texture a screen leaves behind. The **color statistics** detect the slight shift in brightness, saturation, and white balance that happens when a photo is taken through a screen. The **sharpness features** measure subtle loss of detail that occurs because the image passes through two optical systems instead of one. All four features are combined into a 50-number vector and fed into an SVM classifier that outputs a score from 0 (real) to 1 (screen).

**Accuracy:** 93.1 % on a held-out test set of 29 images. The model never falsely flags a real photo as fraud.

**Latency:** ~400 ms per image on CPU (model loaded in memory, pure inference time — no disk I/O per request).

**Cost:** ~$2–$5 per million images on a CPU cloud instance (AWS t3.medium at ~$0.04/hr, model kept loaded in memory between requests, ~2.5–5 images/sec depending on whether both cores are used in parallel). No GPU required. On-device deployment costs nothing.

---

## Files

| File | Purpose |
|---|---|
| `predict.py` | All feature extraction + inference in one file |
| `train.py` | Training pipeline — split, augment, grid-search SVM |
| `run.py` | Streamlit GUI — single image and bulk upload |
| `model.joblib` | Trained SVM model (~81 KB) |
| `requirements.txt` | Python dependencies |

---

## Run

```bash
pip install -r requirements.txt
```

**CLI:**
```bash
python predict.py path/to/image.jpg
```

**GUI:**
```bash
streamlit run run.py
```

**Retrain:**
```bash
python train.py --real real/ --screen screen/ --n_aug 7
```






