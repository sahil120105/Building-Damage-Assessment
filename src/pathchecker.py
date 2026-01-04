import os
import glob

# These must match exactly what is in your train.py
RAW_IMG_DIR = "../data/raw/xbd/tier1/images"
MASK_DIR = "../data/processed/train_masks"

def check_paths():
    print(f"Current Working Directory: {os.getcwd()}")
    
    # 1. Check Raw Images
    print(f"\nChecking Raw Images in: {os.path.abspath(RAW_IMG_DIR)}")
    if not os.path.exists(RAW_IMG_DIR):
        print("❌ ERROR: The raw image directory does not exist!")
    else:
        images = glob.glob(os.path.join(RAW_IMG_DIR, "*_pre_disaster.png"))
        print(f"✅ Found {len(images)} pre-disaster images.")
        if len(images) > 0:
            print(f"   Sample: {images[0]}")

    # 2. Check Masks
    print(f"\nChecking Masks in: {os.path.abspath(MASK_DIR)}")
    if not os.path.exists(MASK_DIR):
        print("❌ ERROR: The mask directory does not exist! Did you run preprocess.py?")
    else:
        masks = glob.glob(os.path.join(MASK_DIR, "*.png"))
        print(f"✅ Found {len(masks)} masks.")
        if len(masks) > 0:
            print(f"   Sample: {masks[0]}")

    # 3. Check Matching
    if len(images) > 0 and len(masks) > 0:
        print("\nChecking Pair Matching...")
        img_name = os.path.basename(images[0])
        # Logic from dataloader: pre_disaster -> post_disaster
        expected_mask = img_name.replace("_pre_disaster.png", "_post_disaster.png")
        mask_path = os.path.join(MASK_DIR, expected_mask)
        
        if os.path.exists(mask_path):
            print(f"✅ SUCCESS: Matched {img_name} with {expected_mask}")
        else:
            print(f"❌ ERROR: Could not find mask for {img_name}")
            print(f"   Expected at: {mask_path}")

if __name__ == "__main__":
    check_paths()