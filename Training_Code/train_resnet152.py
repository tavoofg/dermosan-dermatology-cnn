import os
import random
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ── 1) Checkpoints directory ───────────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)

# ── 2) Seed for reproducibility ────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── 3) Paths and parameters ───────────────────────────────────────────────────
DATA_DIR   = "IMG_CLASSES"
BATCH_SIZE = 32
IMG_SIZE   = (224, 224)
AUTOTUNE   = tf.data.AUTOTUNE

# ── 4) Clases ───────────────────────────────────────────────────────────────
classes     = sorted(os.listdir(DATA_DIR))
class_index = {name: i for i, name in enumerate(classes)}
print("Classes found:", class_index)

# ── 5) Preprocessing ─────────────────────────────────────────────────────────
from tensorflow.keras.applications.resnet import preprocess_input

def process_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    return img, tf.cast(label, tf.int32)

def make_dataset(pairs):
    files  = [p for p, _ in pairs]
    labels = [class_index[l] for _, l in pairs]
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(process_image, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(1000, seed=SEED).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ── 6) Split train/val/test ─────────────────────────────────────────────────
train_pairs, val_pairs, test_pairs = [], [], []
for cls in classes:
    imgs    = os.listdir(os.path.join(DATA_DIR, cls))
    n_train = int(0.8 * len(imgs))
    n_val   = int(0.1 * len(imgs))
    train   = random.sample(imgs, n_train)
    rest    = list(set(imgs) - set(train))
    val     = random.sample(rest, n_val)
    test    = list(set(rest) - set(val))
    for f in train: train_pairs.append((os.path.join(DATA_DIR, cls, f), cls))
    for f in val:   val_pairs.append(  (os.path.join(DATA_DIR, cls, f), cls))
    for f in test:  test_pairs.append( (os.path.join(DATA_DIR, cls, f), cls))

random.shuffle(train_pairs)
random.shuffle(val_pairs)
random.shuffle(test_pairs)
train_ds = make_dataset(train_pairs)
val_ds   = make_dataset(val_pairs)
test_ds  = make_dataset(test_pairs)

# ── 7) Class weights ───────────────────────────────────────────────────────
train_labels = np.concatenate([y for _, y in train_ds], axis=0)
weights      = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(weights))
print("Class weights:", class_weights)

# ── 8) Model ───────────────────────────────────────────────────────────────
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

base = ResNet152(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
for layer in base.layers[:-50]:
    layer.trainable = False

x   = GlobalAveragePooling2D()(base.output)
x   = Dense(512, activation='relu')(x)
x   = Dropout(0.35)(x)
x   = Dense(256, activation='relu')(x)
out = Dense(len(classes), activation='softmax')(x)
model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── 9) Callbacks ────────────────────────────────────────────────────────────
lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

checkpoint_cb = ModelCheckpoint(
    filepath="checkpoints/resnet152-epoch{epoch:02d}-val{val_loss:.2f}.weights.h5",
    save_weights_only=True,
    save_freq="epoch",
    verbose=1
)

# ── 10) (Optional) Resume from a checkpoint ───────────────────────────────
# 🔹 If this is your first time training, KEEP the following lines COMMENTED.
# 🔹 If you already have a checkpoint (.h5) and want to continue training,
#    uncomment and adjust the path + epoch number as needed.

# last_ckpt = "checkpoints/resnet152-epoch03-val0.69.weights.h5"
# model.load_weights(last_ckpt)
# initial_epoch = 3     # Epoch number where training was stopped
initial_epoch = 0        # 0 if starting from scratch
total_epochs  = 9        # Total epochs to train


# ── 11) Training ───────────────────────────────────────────────────────
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,          # Total desired epochs (e.g., 9)
    initial_epoch=initial_epoch,  # If resuming, adjust this value
    class_weight=class_weights,
    callbacks=[lr_cb, checkpoint_cb]
)


# ── 12) Save final model ────────────────────────────────────────────────
model.save("best_resnet152.h5")

# ── 13) Evaluate on test set ─────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true, y_pred = [], []
for imgs, labs in test_ds:
    preds = model.predict(imgs)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labs.numpy())
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.show()

# ── 14) Plots de pérdida y precisión ────────────────────────────────────────
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val   Loss')
plt.legend(); plt.show()
plt.plot(history.history['accuracy'],     label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val   Acc')
plt.legend(); plt.show()
