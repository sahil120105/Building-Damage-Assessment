import os
import json
import cv2
import numpy as np
import glob
from tqdm import tqdm
from shapely import wkt
import sys

# --- CONFIGURATION ---
# Adjust these paths based on where you extracted the data
RAW_DATA_DIR = "../data/raw/xbd/tier1" 
OUTPUT_DIR = "../data/processed/train_masks"

IMG_WIDTH = 1024
IMG_HEIGHT = 1024

# The exact class mapping described in your guide
# 0 is background (black), 1-4 are damage levels
DAMAGE_MAP = {
    "background": 0,
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1  # Treat unclassified as no-damage (standard practice)
}

def parse_json(json_path):
    """
    Helper to load a JSON file safely.
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

def get_geometry_from_pre(pre_json):
    """
    Extracts {uid: polygon_coords} from PRE-disaster JSON.
    We trust PRE for the "where".
    """
    geom_map = {}
    
    # xBD JSON structure: data['features']['xy'] contains list of buildings
    features = pre_json.get('features', {}).get('xy', [])
    
    for feat in features:
        uid = feat['properties']['uid']
        wkt_str = feat['wkt']
        
        # Parse WKT to get integer coordinates
        # Shapely handles the "POLYGON ((x y, ...))" parsing reliably
        try:
            poly = wkt.loads(wkt_str)
            # Extract exterior coordinates as a list of integer points
            coords = list(poly.exterior.coords)
            # Convert to numpy array of int32 for OpenCV
            pts = np.array(coords, np.int32)
            geom_map[uid] = pts
        except Exception as e:
            # Skip invalid polygons
            continue
            
    return geom_map

def get_damage_from_post(post_json):
    """
    Extracts {uid: damage_int} from POST-disaster JSON.
    We trust POST for the "what happened".
    """
    damage_map = {}
    
    features = post_json.get('features', {}).get('xy', [])
    
    for feat in features:
        uid = feat['properties']['uid']
        subtype = feat['properties'].get('subtype', 'no-damage')
        
        # Convert string label to integer
        damage_val = DAMAGE_MAP.get(subtype, 0)
        damage_map[uid] = damage_val
        
    return damage_map

def generate_mask(pre_path, post_path, output_path):
    """
    Combines PRE geometry and POST damage to create the mask.
    """
    pre_data = parse_json(pre_path)
    post_data = parse_json(post_path)
    
    if not pre_data or not post_data:
        return

    # 1. Build Lookups
    # Map: UID -> Polygon (from PRE)
    geometry_map = get_geometry_from_pre(pre_data)
    # Map: UID -> Damage Class (from POST)
    damage_label_map = get_damage_from_post(post_data)
    
    # 2. Initialize blank mask (Zeros = Background)
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    
    # 3. Draw Buildings
    # Iterate through buildings found in PRE (since that defines valid geometry)
    for uid, polygon in geometry_map.items():
        
        # Find corresponding damage in POST
        # If a building exists in PRE but is missing in POST JSON, 
        # it usually implies 'no-damage' or data mismatch. Default to 1 (no-damage).
        damage_class = damage_label_map.get(uid, 1)
        
        # Draw filled polygon
        # color=damage_class will set the pixel values inside the building
        cv2.fillPoly(mask, [polygon], color=damage_class)
        
    # 4. Save Mask
    cv2.imwrite(output_path, mask)

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all PRE-disaster JSONs
    # Pattern matches: .../tier1/labels/*_pre_disaster.json
    search_path = os.path.join(RAW_DATA_DIR, "labels", "*_pre_disaster.json")
    pre_files = glob.glob(search_path)
    
    print(f"Found {len(pre_files)} scenes to process in {RAW_DATA_DIR}...")
    
    if len(pre_files) == 0:
        print("No files found! Check your RAW_DATA_DIR path.")
        sys.exit()

    for pre_file in tqdm(pre_files):
        # Infer POST filename (files are always paired)
        # "guatemala-volcano_00000000_pre_disaster.json" -> "guatemala-volcano_00000000_post_disaster.json"
        post_file = pre_file.replace("_pre_disaster.json", "_post_disaster.json")
        
        if not os.path.exists(post_file):
            # Skip if the pair is incomplete
            continue
            
        # Define Output Filename
        # We save it as the "post" filename but .png, because the mask represents the post-state
        filename = os.path.basename(post_file).replace(".json", ".png")
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Generate
        generate_mask(pre_file, post_file, output_path)

if __name__ == "__main__":
    main()