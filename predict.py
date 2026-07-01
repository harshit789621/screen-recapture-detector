import sys
import time
import joblib
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from skimage.feature import local_binary_pattern
MODEL_PATH = "model.joblib"
IMG_SIZE   = 768   
LBP_SIZE   = 256   
LBP_RADIUS = 2
LBP_POINTS = 16   
FEATURE_DIM = 18 + (LBP_POINTS + 2) + 12 + 2  

def _load_gray_and_color(path_or_array):
    if isinstance(path_or_array, str):
        img = cv2.imread(path_or_array)
        if img is None:
            raise ValueError(f"Could not read image: {path_or_array}")

    elif isinstance(path_or_array, np.ndarray):
        img = path_or_array

    else:
        if hasattr(path_or_array, "seek"):
            path_or_array.seek(0)          # reset pointer in case it was read before
        raw = np.frombuffer(path_or_array.read(), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode uploaded image bytes.")

    h, w  = img.shape[:2]
    side  = min(h, w)
    y0    = (h - side) // 2
    x0    = (w - side) // 2
    img   = img[y0:y0 + side, x0:x0 + side]
    img   = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray, img

def fft_features(gray):
    f      = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag    = np.log1p(np.abs(fshift))

    h, w   = mag.shape
    cy, cx = h // 2, w // 2
    Y, X   = np.ogrid[:h, :w]
    r      = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(np.int32)
    max_r  = min(cy, cx)

    edges  = np.linspace(0, max_r, 17)
    profile = []
    for i in range(16):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        profile.append(float(mag[mask].mean()) if mask.sum() > 0 else 0.0)
    profile = np.array(profile)
    profile /= (profile.sum() + 1e-8) 
    hf_ratio = float(mag[r > 0.4 * max_r].sum() / (mag.sum() + 1e-8))

    donut = mag.copy()
    donut[r < 0.05 * max_r] = 0
    n_peaks = int((donut > donut.mean() + 3 * donut.std()).sum())

    return np.array(list(profile) + [hf_ratio, np.log1p(n_peaks)], dtype=np.float32)


def lbp_features(gray):
    small  = cv2.resize(gray, (LBP_SIZE, LBP_SIZE), interpolation=cv2.INTER_AREA)
    lbp    = local_binary_pattern(small, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def color_features(bgr):
    hsv    = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    feats = [s.mean(), s.std(), v.mean(), v.std(), h.mean(), h.std()]
    for c in range(3):                          # per-channel BGR mean + std
        ch = bgr[..., c].astype(np.float32)
        feats += [ch.mean(), ch.std()]

    return np.array(feats, dtype=np.float32)

def sharpness_features(gray):
    lap_var      = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edge_density = float((cv2.Canny(gray, 50, 150) > 0).mean())
    return np.array([np.log1p(lap_var), edge_density], dtype=np.float32)


def extract_features(path_or_array):
    """
    Full pipeline: load → 4 feature families → concatenate.
    Returns a float32 vector of length FEATURE_DIM (50).
    """
    gray, bgr = _load_gray_and_color(path_or_array)

    with ThreadPoolExecutor(max_workers=4) as ex:
        f_fft   = ex.submit(fft_features,       gray)
        f_lbp   = ex.submit(lbp_features,       gray)
        f_color = ex.submit(color_features,      bgr)
        f_sharp = ex.submit(sharpness_features,  gray)

    return np.concatenate([
        f_fft.result(),
        f_lbp.result(),
        f_color.result(),
        f_sharp.result(),
    ])
def augment_bgr(bgr, rng):
    out = bgr.copy()
    angle = rng.uniform(-8, 8)
    h, w  = out.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out   = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    alpha = rng.uniform(0.85, 1.15)
    beta  = rng.uniform(-15, 15)
    out   = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    quality = int(rng.integers(60, 95))
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out
_bundle = None

def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(image_path, debug: bool = False) -> float:
    t0 = time.perf_counter()

    bundle        = _get_bundle()
    scaler, clf   = bundle["scaler"], bundle["clf"]

    t1   = time.perf_counter()
    feat = extract_features(image_path)

    t2     = time.perf_counter()
    feat_s = scaler.transform([feat])

    t3    = time.perf_counter()
    score = float(clf.predict_proba(feat_s)[0][1])

    t4 = time.perf_counter()

    if debug:
        print(
            f"load={1000*(t1-t0):.0f}ms  extract={1000*(t2-t1):.0f}ms  "
            f"scale={1000*(t3-t2):.0f}ms  predict={1000*(t4-t3):.0f}ms",
            file=sys.stderr,
        )

    return score
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    t0    = time.perf_counter()
    score = predict(sys.argv[1], debug=True)
    t1    = time.perf_counter()

    print(f"{score:.4f}")
    print(f"(total wall-clock latency: {(t1 - t0) * 1000:.1f} ms)", file=sys.stderr)