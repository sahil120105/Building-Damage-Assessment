import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import random

# --- CONFIGURATION ---
MODEL_PATH = "../results/models/xbd_model_best.keras"
TEST_IMG_DIR = "../data/raw/xbd/tier1/images" # Using train images for demo if test is empty
INPUT_SHAPE = (512, 512)

# Visualization Colors (Same as preprocess)
# Format: RGB (Red, Green, Blue) for Matplotlib
DAMAGE_COLORS = {
    0: [0, 0, 0],       # Background (Black)
    1: [0, 255, 0],     # No Damage (Green)
    2: [255, 255, 0],   # Minor Damage (Yellow)
    3: [255, 165, 0],   # Major Damage (Orange)
    4: [255, 0, 0]      # Destroyed (Red)
}

def decode_mask(mask_int):
    """
    Converts a (512, 512, 1) Integer Mask -> (512, 512, 3) RGB Image
    """
    h, w, _ = mask_int.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Remove single channel dimension for easier iteration
    mask_flat = mask_int.squeeze() 
    
    for class_id, color in DAMAGE_COLORS.items():
        # Where the pixel value equals class_id, set the color
        mask_rgb[mask_flat == class_id] = color
        
    return mask_rgb

def predict_sample(model, pre_path, post_path):
    # 1. Load Images
    img_pre_orig = cv2.imread(pre_path)
    img_post_orig = cv2.imread(post_path)
    
    # 2. Preprocess for Model (Resize + Normalize)
    # Model expects (Batch, 512, 512, 3)
    img_pre = cv2.resize(img_pre_orig, INPUT_SHAPE)
    img_post = cv2.resize(img_post_orig, INPUT_SHAPE)
    
    # Convert BGR (OpenCV) to RGB (Model)
    img_pre_rgb = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB) / 255.0
    img_post_rgb = cv2.cvtColor(img_post, cv2.COLOR_BGR2RGB) / 255.0
    
    # Add Batch Dimension
    input_pre = np.expand_dims(img_pre_rgb, axis=0)
    input_post = np.expand_dims(img_post_rgb, axis=0)
    
    # 3. Predict
    print(f"Running inference on {os.path.basename(pre_path)}...")
    prediction = model.predict([input_pre, input_post]) # Shape: (1, 512, 512, 5)
    
    # 4. Post-process
    # Take the highest probability class for each pixel
    pred_mask_int = np.argmax(prediction, axis=-1) # Shape: (1, 512, 512)
    pred_mask_int = np.expand_dims(pred_mask_int[0], axis=-1) # Back to (512, 512, 1)
    
    # Convert Integers to RGB Colors
    pred_mask_rgb = decode_mask(pred_mask_int)
    
    # 5. Create Overlay (Blend Prediction with Post-Image)
    # We use the resized post-image for blending
    img_post_resized = cv2.cvtColor(img_post, cv2.COLOR_BGR2RGB) # Keep as uint8 (0-255)
    
    # Blend: 70% Original Image + 30% Mask
    # Note: Background (Black) in mask will darken the image slightly, 
    # but colored buildings will pop out.
    overlay = cv2.addWeighted(img_post_resized, 0.7, pred_mask_rgb, 0.3, 0)
    
    return img_pre_rgb, img_post_rgb, pred_mask_rgb, overlay

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. Did you finish training?")
        return

    print("Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Pick a random pair from the raw folder
    all_pre_images = glob(os.path.join(TEST_IMG_DIR, "*_pre_disaster.png"))
    
    if len(all_pre_images) == 0:
        print("No images found to test.")
        return

    # Select 3 random samples
    samples = random.sample(all_pre_images, 3)
    
    for pre_path in samples:
        post_path = pre_path.replace("_pre_disaster.png", "_post_disaster.png")
        
        if not os.path.exists(post_path):
            continue
            
        # Run Prediction
        img_pre, img_post, prediction, overlay = predict_sample(model, pre_path, post_path)
        
        # Plotting
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.title("Pre-Disaster")
        plt.imshow(img_pre)
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.title("Post-Disaster")
        plt.imshow(img_post)
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.title("Damage Prediction")
        plt.imshow(prediction)
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title("Overlay (Context)")
        plt.imshow(overlay)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()