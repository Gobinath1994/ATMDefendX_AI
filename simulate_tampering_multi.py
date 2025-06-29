"""
ATM Tampering Image Generator using OpenCV
------------------------------------------

This script takes clean ATM images and programmatically applies various tampering effects
to simulate real-world fraud scenarios such as:

- Fake card readers
- Hidden camera dots
- Security overlay stickers
- Keypad overlays

It generates multiple combinations of tampering per image and saves them in a separate folder
for training or evaluation purposes.

Output: tampered images saved in data/tampered_atms/
"""

import os
import cv2
import random
import numpy as np

# === Directory paths ===
clean_dir = "data/clean_atms"            # Folder with clean reference ATM images
tampered_dir = "data/tampered_atms"      # Output folder for tampered images
os.makedirs(tampered_dir, exist_ok=True)

# === Tampering Simulations ===

def simulate_fake_reader(img):
    """Draws a fake card reader rectangle on the ATM image."""
    h, w = img.shape[:2]
    img_copy = img.copy()
    start_point = (int(0.58 * w), int(0.42 * h))
    end_point = (int(0.92 * w), int(0.52 * h))
    cv2.rectangle(img_copy, start_point, end_point, (30, 30, 30), -1)
    cv2.putText(img_copy, "FAKE READER", (start_point[0], start_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img_copy

def simulate_camera_dot(img):
    """Draws a small red circle to simulate a hidden camera."""
    h, w = img.shape[:2]
    img_copy = img.copy()
    center = (random.choice([20, w - 20]), random.choice([20, h - 20]))
    cv2.circle(img_copy, center, 8, (0, 0, 255), -1)
    return img_copy

def simulate_overlay_sticker(img):
    """Adds a translucent yellow security sticker on top of the image."""
    h, w = img.shape[:2]
    img_copy = img.copy()
    overlay = img.copy()
    start = (int(0.15 * w), int(0.15 * h))
    end = (int(0.35 * w), int(0.25 * h))
    cv2.rectangle(overlay, start, end, (255, 255, 0), -1)
    cv2.putText(overlay, "SECURITY ALERT", (start[0]+5, start[1]+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return cv2.addWeighted(overlay, 0.7, img_copy, 0.3, 0)

def simulate_keypad_overlay(img):
    """Adds a grey keypad overlay rectangle with label."""
    h, w = img.shape[:2]
    img_copy = img.copy()
    overlay = img.copy()
    start = (int(0.42 * w), int(0.72 * h))
    end = (int(0.65 * w), int(0.92 * h))
    cv2.rectangle(overlay, start, end, (100, 100, 100), -1)
    cv2.putText(overlay, "OVERLAY", (start[0]+5, start[1]+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.addWeighted(overlay, 0.5, img_copy, 0.5, 0)

# === Tampering Function Pool ===
tamper_functions = [
    ("reader", simulate_fake_reader),
    ("camera", simulate_camera_dot),
    ("sticker", simulate_overlay_sticker),
    ("keypad", simulate_keypad_overlay),
]

def apply_combo(img, funcs):
    """
    Apply a combination of tampering functions sequentially.
    
    Args:
        img (np.array): Original image
        funcs (list): List of tampering functions

    Returns:
        np.array: Tampered image
    """
    tampered = img.copy()
    for func in funcs:
        tampered = func(tampered)
    return tampered

# === Main Loop: Apply tampering to all clean images ===
for filename in os.listdir(clean_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(clean_dir, filename)
        img = cv2.imread(img_path)
        base_name = os.path.splitext(filename)[0]

        # Generate 4 variants per clean image with 1–3 tampering types
        for i in range(1, 5):
            selected = random.sample(tamper_functions, k=random.randint(1, 3))
            suffix = "_".join([name for name, _ in selected])
            funcs = [func for _, func in selected]
            tampered_img = apply_combo(img, funcs)
            out_name = f"{base_name}_{suffix}_{i}.jpg"
            cv2.imwrite(os.path.join(tampered_dir, out_name), tampered_img)

print("✅ Multiple tampered ATM images generated in:", tampered_dir)