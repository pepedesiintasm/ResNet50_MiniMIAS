import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ======================
# CONFIG
# ======================
MODEL_PATH = "../models/resnet50_cbis_best.h5"
IMAGE_PATH = "/Users/pepedesintas/Desktop/TFG/CBIS_DDSM/processed/test/malignant/1-019.jpg"
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "conv5_block3_out"

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(MODEL_PATH)

# ======================
# LOAD & PREPROCESS IMAGE
# ======================
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    img_array = np.expand_dims(img, axis=0).astype("float32")
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    return img, img_array

# ======================
# GRAD-CAM
# ======================
def make_gradcam_heatmap(img_array, model, last_conv_layer):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# ======================
# OVERLAY
# ======================
def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ======================
# RUN
# ======================
orig_img, img_array = load_image(IMAGE_PATH)
heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
overlay = overlay_heatmap(orig_img, heatmap)

# ======================
# PLOT
# ======================
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(orig_img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grad-CAM")
plt.imshow(heatmap, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()
