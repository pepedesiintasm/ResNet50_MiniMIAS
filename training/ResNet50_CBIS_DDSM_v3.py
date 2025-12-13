import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ======================
# CONFIGURACIÓN
# ======================
DATA_DIR = "/Users/pepedesintas/Desktop/TFG/CBIS_DDSM/processed"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# ======================
# DATASETS
# ======================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "valid"),
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Clases:", train_ds.class_names)  # ['benign', 'malignant']

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ======================
# DATA AUGMENTATION
# ======================
# Moderado, adecuado para CBIS-DDSM
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.15),
])

# ======================
# MODELO
# ======================
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)

base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=x
)

# ======================
# FASE 1: FEATURE EXTRACTOR
# ======================
# Congelar TODA la ResNet
base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# ======================
# COMPILACIÓN FASE 1
# ======================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ======================
# CALLBACKS FASE 1
# ======================
callbacks_phase1 = [
    EarlyStopping(
        monitor="val_auc",
        patience=3,
        mode="max",
        restore_best_weights=True
    )
]

# ======================
# ENTRENAMIENTO FASE 1
# ======================
print("\n🔹 FASE 1: Entrenando solo la cabeza")
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks_phase1
)

# ======================
# FASE 2: FINE-TUNING SUAVE
# ======================
# Congelar todo
for layer in base_model.layers:
    layer.trainable = False

# Descongelar SOLO el último bloque (conv5_x)
for layer in base_model.layers[-30:]:
    layer.trainable = True

# ======================
# COMPILACIÓN FASE 2
# ======================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# ======================
# CALLBACKS FASE 2
# ======================
callbacks_phase2 = [
    EarlyStopping(
        monitor="val_auc",
        patience=4,
        mode="max",
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "../models/resnet50_cbis_bestV3.h5",
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
]

# ======================
# ENTRENAMIENTO FASE 2
# ======================
print("\n🔹 FASE 2: Fine-tuning del último bloque")
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks_phase2
)

# ======================
# EVALUACIÓN FINAL
# ======================
print("\n🧪 Evaluación en test:")
model.evaluate(test_ds)

# ======================
# GUARDADO FINAL
# ======================
model.save("../models/resnet50_cbis_finalV3.h5")
print("✅ Modelo CBIS-DDSM guardado correctamente")
