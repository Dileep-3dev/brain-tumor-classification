"""
Trains the small model that ships with this repository.

Why this exists alongside training.py
-------------------------------------
training.py builds a 224x224 network ending in Flatten -> Dense(512): roughly
19 million parameters in that single layer, about 75 MB on disk, and hours per
run on a CPU. Too large to commit and too slow to reproduce.

This script makes two changes:

  1. GlobalAveragePooling2D instead of Flatten, which removes ~18.9M
     parameters and leaves a ~2 MB model that fits in git.

  2. Images are decoded once into an in-memory array and cached to disk.
     ImageDataGenerator re-reads and re-decodes all 8,582 JPEGs every epoch,
     and on a laptop that disk work - not the maths - is what makes training
     slow. Loading once turns minutes per epoch into seconds.

Validation is split out of the training images, so the test directory is read
exactly once, at the very end.

Run:  python train_compact.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image

IMG = (96, 96)
BATCH = 64
EPOCHS = 25
SEED = 42
OUT = Path("models")
CACHE = Path(".cache_images.npz")
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def load_split(root):
    """Decode every image under root/<class>/ into one uint8 array."""
    images, labels = [], []
    for idx, name in enumerate(CLASSES):
        folder = Path(root) / name
        if not folder.is_dir():
            raise SystemExit(f"Missing folder: {folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for path in files:
            with Image.open(path) as im:
                images.append(np.asarray(im.convert("RGB").resize(IMG), dtype=np.uint8))
            labels.append(idx)
        print(f"   {name:20} {len(files):5d}")
    return np.stack(images), np.array(labels, dtype=np.int64)


def load_all():
    if CACHE.exists():
        print(f"Loading cached images from {CACHE}")
        d = np.load(CACHE)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    print("Decoding training images (one time, then cached):")
    xtr, ytr = load_split("data/train")
    print("Decoding test images:")
    xte, yte = load_split("data/test")
    np.savez_compressed(CACHE, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    print(f"Cached to {CACHE} ({CACHE.stat().st_size / 1024 / 1024:.0f} MB)")
    return xtr, ytr, xte, yte


def main():
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    xtr_all, ytr_all, xte, yte = load_all()
    # Validation is carved out of the TRAINING images; the test set stays untouched.
    xtr, xval, ytr, yval = train_test_split(
        xtr_all, ytr_all, test_size=0.2, stratify=ytr_all, random_state=SEED
    )
    # Scale to [0,1] here, matching what the Streamlit app sends at inference.
    xtr = xtr.astype("float32") / 255.0
    xval = xval.astype("float32") / 255.0
    xte_f = xte.astype("float32") / 255.0
    print(f"\ntrain {len(ytr)}   val {len(yval)}   test {len(yte)}")

    weights = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    class_weight = dict(enumerate(weights))
    print("class weights:", {k: round(v, 3) for k, v in class_weight.items()})

    def block(f):
        # BatchNormalization momentum is lowered from the 0.99 default to 0.9.
        # With only ~108 steps per epoch, 0.99 leaves the moving mean/variance
        # far behind the true statistics, and because those statistics are
        # gathered on augmented images the mismatch at inference collapsed
        # every validation prediction onto a single class.
        return [layers.Conv2D(f, 3, activation="relu", padding="same"),
                layers.BatchNormalization(momentum=0.9), layers.MaxPooling2D(2)]

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*IMG, 3)),
        # No Rescaling layer here on purpose: main.py and training.py both
        # hand the model images already scaled to [0,1], so rescaling again
        # inside the model would divide by 255 twice.
        # Augmentation lives in the model, so it is active during training and
        # automatically bypassed at inference.
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
        layers.RandomZoom(0.08, seed=SEED),
        layers.RandomTranslation(0.05, 0.05, seed=SEED),
        *block(32), *block(64), *block(128), *block(256),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(f"trainable parameters: {model.count_params():,}\n")

    OUT.mkdir(exist_ok=True)
    ckpt = OUT / "brain_tumor_cnn.keras"
    model.fit(
        xtr, ytr, validation_data=(xval, yval), epochs=EPOCHS, batch_size=BATCH,
        class_weight=class_weight, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7,
                                             restore_best_weights=True, mode="max"),
            tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy",
                                               save_best_only=True, mode="max"),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=2, min_lr=1e-5),
        ],
    )

    # The test set is touched exactly once, now that training and model
    # selection are both finished.
    loss, acc = model.evaluate(xte_f, yte, verbose=0)
    pred = model.predict(xte_f, verbose=0).argmax(1)
    print(f"\nHeld-out test accuracy: {acc:.4f}   loss: {loss:.4f}")

    print("\nPer-class recall:")
    per_class = {}
    for i, name in enumerate(CLASSES):
        m = yte == i
        r = float((pred[m] == i).mean()) if m.sum() else 0.0
        per_class[name] = round(r, 4)
        print(f"   {name:20} {r:.4f}  (n={int(m.sum())})")

    cm = [[int(((yte == a) & (pred == b)).sum()) for b in range(len(CLASSES))]
          for a in range(len(CLASSES))]
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print("   " + "".join(f"{c[:8]:>10}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"   {CLASSES[i][:8]:>8}" + "".join(f"{v:>10}" for v in row))

    model.save(ckpt)
    size_mb = ckpt.stat().st_size / 1024 / 1024
    (OUT / "model_card.json").write_text(json.dumps({
        "input_size": list(IMG), "classes": CLASSES,
        "parameters": int(model.count_params()),
        "test_accuracy": round(float(acc), 4), "test_loss": round(float(loss), 4),
        "per_class_recall": per_class, "confusion_matrix": cm,
        "train_images": int(len(ytr)), "val_images": int(len(yval)),
        "test_images": int(len(yte)), "size_mb": round(size_mb, 2),
        "note": "Demo model at 96x96. training.py holds the full 224x224 design.",
    }, indent=2))
    print(f"\nSaved {ckpt} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
