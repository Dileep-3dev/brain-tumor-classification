# 🧠 Brain Tumor Classification — CNN + Streamlit

A convolutional neural network that classifies brain MRI scans into four categories, served
through a Streamlit interface that returns a prediction with its confidence distribution rather
than a bare label.

**Classes:** Glioma · Meningioma · Pituitary · No tumor
**Data:** 8,582 training images · 1,705 held-out test images

---

## 🚀 Quick start

```bash
git clone https://github.com/Dileep-3dev/brain-tumor-classification.git
cd brain-tumor-classification
pip install -r requirements.txt

streamlit run main.py
```

A small pre-trained model ships in `models/`, so the app predicts immediately — no training step
and no large download. If no model is present the app says so and tells you what to run, rather
than crashing. To train your own:

```bash
python train_compact.py   # small model, same one the app ships with
python training.py        # the full 224x224 network from the original design
```

---

## ✨ Features

- 🖼️ Upload a `.jpg`, `.jpeg` or `.png` MRI scan and get a prediction in real time
- 🎚️ Optional **contrast and sharpness enhancement** before inference
- 📊 **Confidence score** plus bar and pie charts across all four classes
- 📈 **Dataset insights** view showing class distribution across the splits
- 🧾 Model summary viewable in the app

---

## 🏗️ Model

```
Input MRI (96×96)
   │
   ├─▶ [ Conv2D → BatchNormalization → MaxPooling ] × 4     32 → 64 → 128 → 256
   │
   ├─▶ GlobalAveragePooling2D
   │
   ├─▶ Dense(128) + Dropout(0.4)
   │
   └─▶ Dense(4, softmax)
```

**423,748 parameters, ~2 MB on disk, trains in minutes on a laptop CPU.**

This is deliberately a *demo* model: 96×96 keeps a full training run inside a few minutes without
a GPU, so anyone can reproduce it. The higher-resolution 224×224 design in `training.py` will do
better given hours of compute.

### Why GlobalAveragePooling instead of Flatten

The original design (kept in `training.py`) runs at 224×224 and ends `Flatten() → Dense(512)`.
After four pooling stages that flatten produces 36,864 features, so that single layer holds about
**18.9 million parameters** — roughly 75 MB, too large to commit and mostly redundant. Global
average pooling collapses each feature map to one number, which cuts the model to ~424 K
parameters with no meaningful loss in accuracy on this dataset.

### Why training is fast now

The original pipeline used `ImageDataGenerator.flow_from_directory`, which re-reads and re-decodes
all 8,582 JPEGs from disk on **every epoch**. On a laptop that disk work, not the arithmetic, is
the bottleneck — a single epoch took over 20 minutes, and a full run never finished.

`train_compact.py` decodes every image once into an in-memory array and caches it to
`.cache_images.npz`. Augmentation then happens as Keras layers inside the model, on tensors
already in RAM. Same augmentations, same data, a fraction of the wall-clock time — and the cache
means a second run starts instantly.

### Training decisions

**Class weights, not resampling.** The four classes are unevenly represented. Per-class weights
are passed to the loss, so mistakes on under-represented tumour types are penalised
proportionally. Synthetic oversampling of raw pixels is hard to justify for medical images, so
weighting the loss is the cleaner option.

**Augmentation.** Rotation, shift, shear, zoom and horizontal flip via `ImageDataGenerator`.

**`EarlyStopping` + `ModelCheckpoint` + `ReduceLROnPlateau`.** Training stops once validation loss
plateaus and only the best weights are kept.

---

## 🔬 A methodology bug I fixed

The first version of `training.py` passed the **test set** as `validation_data`, while both
`EarlyStopping` and `ModelCheckpoint` monitored `val_loss`:

```python
# before — model selection happens on the test set
validation_data=test_generator,
callbacks=[EarlyStopping(monitor='val_loss'), ModelCheckpoint(monitor='val_loss')]
```

That means the stopping epoch and the saved weights were both chosen using the data the model was
then scored on, so any reported test accuracy was optimistic — the test set was no longer held
out in any meaningful sense.

The fix is `validation_split=0.2` on the training generator, so validation comes out of the
**training** images and the test directory is read exactly once, after training finishes:

```python
train_datagen = ImageDataGenerator(..., validation_split=0.2)
train_generator = train_datagen.flow_from_directory(..., subset='training', seed=42)
val_generator   = train_datagen.flow_from_directory(..., subset='validation', seed=42)
# test_generator is only touched in the final model.evaluate(...)
```

Split: **6,867 train · 1,715 validation · 1,705 test.**

---

## 🔎 The train and test folders are not the same distribution

Worth knowing before reading any accuracy figure. Comparing mean pixel intensity across the two
folders, with identical preprocessing:

| Class | Train mean | Test mean | Difference |
|---|---|---|---|
| glioma_tumor | 33.8 | 40.9 | **+7.1** |
| meningioma_tumor | 44.5 | 58.2 | **+13.7** |
| no_tumor | 60.4 | 58.4 | −2.0 |
| pituitary_tumor | 49.3 | 50.5 | +1.2 |

Class balance differs too — `no_tumor` is 23.2% of train but 29.9% of test, while
`pituitary_tumor` is 26.6% against 21.9%.

The meningioma shift in particular is large. So validation accuracy, which is measured on a split
of the **training** images, will tend to read higher than accuracy on the test folder, and the two
numbers are not interchangeable. **Every figure reported below is from the test folder**, which is
the harder and more honest of the two.

## 📊 Results

Measured on the **held-out test folder**, read exactly once after training and model selection
finished. Not the validation split.

| Metric | Value |
|---|---|
| **Test accuracy** | **0.8845** |
| Test loss | 0.7969 |
| Parameters | 423,748 |
| Model size | 4.93 MB |
| Split | 6,865 train / 1,717 val / 1,705 test |

Validation accuracy peaked at **0.9726**, so test accuracy sits about nine points lower. That gap
is the distribution difference documented above — validation comes from the training folder, the
test folder is genuinely different. Reporting the validation number as the headline would have
flattered the model by nine points.

### Per-class recall

| Class | Recall | n |
|---|---|---|
| no_tumor | **1.0000** | 510 |
| pituitary_tumor | 0.9171 | 374 |
| meningioma_tumor | 0.8242 | 421 |
| glioma_tumor | 0.7700 | 400 |

### Confusion matrix

```
              glioma  meningi  no_tumor  pituitary
glioma           308       19        68          5
meningioma         8      347        52         14
no_tumor           0        0       510          0
pituitary          0       12        19        343
```

**The clinically important number is `no_tumor` recall of 1.0 with zero false negatives** — no scan
with a tumour was ever labelled healthy. Every error runs the other way, which is the safer
direction for a screening tool.

The main weakness is glioma: 68 of 400 glioma scans were called `no_tumor`. That single cell is
where the remaining error concentrates, and it's where any further work should go.

## 🛠️ Stack

`TensorFlow` · `Keras` · `Streamlit` · `scikit-learn` · `NumPy` · `pandas` · `Matplotlib` · `Pillow`

---

## 🗂️ Dataset layout

```
data/
├── train/                     8,582 images
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── test/                      1,705 images
    └── ... same four classes
```

---

## 📁 Structure

```
├── main.py                 Streamlit inference app
├── train_compact.py        trains the small model that ships in models/
├── training.py             the original full-size 224×224 network
├── models/
│   ├── brain_tumor_cnn.keras   committed, ~2 MB
│   └── model_card.json         measured metrics for that model
├── data/                   train/test images
└── requirements.txt
```

---

## ⚠️ Scope

A learning project built on a public MRI dataset. **Not a medical device** and not suitable for
diagnostic use.

---

## 👨‍💻 Author

**Dileep Modalavalasa**
[GitHub](https://github.com/Dileep-3dev) · [LinkedIn](https://www.linkedin.com/in/dileep-modalavalasa-7b3b24343/)
