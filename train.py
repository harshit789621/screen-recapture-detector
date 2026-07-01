import argparse
import os
import time
import glob
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from features import extract_features, augment_bgr

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def list_paths(folder):
    return [p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(VALID_EXT)]

def extract_with_augmentation(paths, label, n_aug, rng):
    X, y, used_paths = [], [], []
    for p in paths:
        try:
            X.append(extract_features(p))
            y.append(label)
            used_paths.append(p)
        except Exception as e:
            print(f"  [skip] {p}: {e}")
            continue

        if n_aug > 0:
            img = cv2.imread(p)
            if img is None:
                continue
            for _ in range(n_aug):
                aug = augment_bgr(img, rng)
                X.append(extract_features(aug))
                y.append(label)
                used_paths.append(p + " (augmented)")
    return X, y, used_paths

def extract_plain(paths, label):
    X, y, used_paths = [], [], []
    for p in paths:
        try:
            X.append(extract_features(p))
            y.append(label)
            used_paths.append(p)
        except Exception as e:
            print(f"  [skip] {p}: {e}")
    return X, y, used_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="real", help="folder of real photos")
    ap.add_argument("--screen", default="screen", help="folder of screen/recapture photos")
    ap.add_argument("--out", default="model.joblib", help="output model file")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--n_aug", type=int, default=4, help="augmented copies per TRAINING image")
    args = ap.parse_args()

    rng = np.random.default_rng(42)

    real_paths = list_paths(args.real)
    screen_paths = list_paths(args.screen)
    print(f"Found {len(real_paths)} real images, {len(screen_paths)} screen images")

    real_train_p, real_test_p = train_test_split(real_paths, test_size=args.test_size, random_state=42)
    screen_train_p, screen_test_p = train_test_split(screen_paths, test_size=args.test_size, random_state=42)

    print(f"\nExtracting + augmenting training images (n_aug={args.n_aug} per image)...")
    X_real_tr, y_real_tr, _ = extract_with_augmentation(real_train_p, 0, args.n_aug, rng)
    X_screen_tr, y_screen_tr, _ = extract_with_augmentation(screen_train_p, 1, args.n_aug, rng)

    print("Extracting test images (no augmentation)...")
    X_real_te, y_real_te, p_real_te = extract_plain(real_test_p, 0)
    X_screen_te, y_screen_te, p_screen_te = extract_plain(screen_test_p, 1)

    X_train = np.array(X_real_tr + X_screen_tr)
    y_train = np.array(y_real_tr + y_screen_tr)
    X_test = np.array(X_real_te + X_screen_te)
    y_test = np.array(y_real_te + y_screen_te)
    p_test = p_real_te + p_screen_te

    print(f"\nTraining set (incl. augmented): {len(X_train)} images")
    print(f"Held-out test set (original only): {len(X_test)} images")

    if len(X_train) < 10 or len(X_test) < 5:
        raise SystemExit("Not enough images found — check your folder paths.")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    param_grid = {
    "C": [1, 3, 10, 30, 100],
    "gamma": ["scale", 0.01, 0.003, 0.001, 0.0003, 0.0001],
}
    base_clf = SVC(kernel="rbf", probability=True, class_weight=None)
    grid = GridSearchCV(base_clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_s, y_train)
    clf = grid.best_estimator_
    print(f"\nBest params from grid search: {grid.best_params_}")
    print(f"Best 5-fold CV accuracy on training set: {grid.best_score_:.3f}")

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nHeld-out test accuracy: {acc:.3f}  ({len(y_test)} test images)")
    print("\nConfusion matrix (rows=actual, cols=predicted) [0=real, 1=screen]:")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + classification_report(y_test, y_pred, target_names=["real", "screen"]))

    wrong = [(p_test[i], y_test[i], y_pred[i]) for i in range(len(y_test)) if y_test[i] != y_pred[i]]
    if wrong:
        print("Misclassified test images:")
        for p, actual, pred in wrong:
            label_map = {0: "real", 1: "screen"}
            print(f"  {p}  actual={label_map[actual]}  predicted={label_map[pred]}")
    sample_path = p_test[0]
    n_runs = 30
    t0 = time.perf_counter()
    for _ in range(n_runs):
        feat = extract_features(sample_path)
        feat_s = scaler.transform([feat])
        _ = clf.predict_proba(feat_s)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / n_runs * 1000
    print(f"\nAverage latency: {avg_ms:.1f} ms/image (CPU, this machine)")
    joblib.dump({"scaler": scaler, "clf": clf}, args.out)
    print(f"\nSaved trained model -> {args.out}")
    print("Now run: python predict.py some_image.jpg")

if __name__ == "__main__":
    main()