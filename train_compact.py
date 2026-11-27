"""
Trains the small model that ships with this repository.

training.py builds a 224x224 network ending in Flatten -> Dense(512), which is
roughly 19 million parameters in that one layer and about 75 MB on disk - too
large to commit. This script keeps the same idea but uses a 160x160 input and
GlobalAveragePooling2D instead of Flatten, giving a model of a few hundred
thousand parameters that fits comfortably in git, so the Streamlit app works
immediately after `git clone`.

As in training.py, validation comes out of the training images and the test
directory is only touched once, at the end.

Run:  python train_compact.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                                     GlobalAveragePooling2D, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG = (160, 160)
BATCH = 32
EPOCHS = 12
OUT = Path("models")
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

train_aug = ImageDataGenerator(
    rescale=1.0 / 255, rotation_range=20, width_shift_range=0.15,
    height_shift_range=0.15, shear_range=0.15, zoom_range=0.15,
    horizontal_flip=True, fill_mode="nearest", validation_split=0.2,
)
plain = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_aug.flow_from_directory("data/train", target_size=IMG, batch_size=BATCH,
                                          class_mode="categorical", subset="training", seed=SEED)
val_gen = train_aug.flow_from_directory("data/train", target_size=IMG, batch_size=BATCH,
                                        class_mode="categorical", subset="validation", seed=SEED)
test_gen = plain.flow_from_directory("data/test", target_size=IMG, batch_size=BATCH,
                                     class_mode="categorical", shuffle=False)

weights = compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weight = dict(enumerate(weights))
print("class weights:", {k: round(v, 3) for k, v in class_weight.items()})


def block(filters):
    return [Conv2D(filters, 3, activation="relu", padding="same"),
            BatchNormalization(), MaxPooling2D(2)]


model = Sequential([
    tf.keras.Input(shape=(*IMG, 3)),
    *block(32), *block(64), *block(128), *block(256),
    GlobalAveragePooling2D(),          # replaces Flatten: ~19M fewer parameters
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(4, activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
print(f"trainable parameters: {model.count_params():,}")

OUT.mkdir(exist_ok=True)
ckpt = OUT / "brain_tumor_cnn.keras"

model.fit(
    train_gen, validation_data=val_gen, epochs=EPOCHS, class_weight=class_weight,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ],
    verbose=2,
)

# The test set is used exactly once, after training and model selection.
loss, acc = model.evaluate(test_gen, verbose=0)
probs = model.predict(test_gen, verbose=0)
pred, true = probs.argmax(1), test_gen.classes
labels = list(test_gen.class_indices)

print(f"\nHeld-out test accuracy: {acc:.4f}   loss: {loss:.4f}")
print("\nPer-class recall:")
per_class = {}
for i, name in enumerate(labels):
    m = true == i
    r = float((pred[m] == i).mean()) if m.sum() else 0.0
    per_class[name] = round(r, 4)
    print(f"   {name:20} {r:.4f}  (n={int(m.sum())})")

model.save(ckpt)
size_mb = ckpt.stat().st_size / 1024 / 1024
(OUT / "model_card.json").write_text(json.dumps({
    "input_size": list(IMG), "classes": labels, "parameters": int(model.count_params()),
    "test_accuracy": round(float(acc), 4), "test_loss": round(float(loss), 4),
    "per_class_recall": per_class, "train_images": int(train_gen.samples),
    "val_images": int(val_gen.samples), "test_images": int(test_gen.samples),
    "size_mb": round(size_mb, 2),
}, indent=2))
print(f"\nSaved {ckpt} ({size_mb:.1f} MB)")
