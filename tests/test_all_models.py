import subprocess
import sys

def check_yolo():
    try:
        from ultralytics import YOLO
        model = YOLO('yoloe-11s-seg.pt')
        print("✅ YOLOv11s: Successfully imported and initialized.")
    except Exception as e:
        print(f"❌ YOLOv11s: FAILED - {e}")

def check_sam2():
    try:
        import torch
        import sam2
        # Just check import for SAM2
        print("✅ SAM2: Successfully imported.")
    except Exception as e:
        print(f"❌ SAM2: FAILED - {e}")

def check_slam3r():
    try:
        result = subprocess.run([sys.executable, "SLAM3R/recon.py", "--help"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ SLAM3R: Successfully executed recon.py --help.")
        else:
            print(f"❌ SLAM3R: FAILED - {result.stderr}")
    except Exception as e:
        print(f"❌ SLAM3R: FAILED - {e}")

def check_colmap():
    try:
        # Check colmap.bat in C:\Users\elite\COLMAP
        result = subprocess.run([r"C:\Users\elite\COLMAP\COLMAP.bat", "-h"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ COLMAP SfM: Successfully executed colmap -h.")
        else:
            print(f"❌ COLMAP SfM: FAILED - {result.stderr}")
    except Exception as e:
        print(f"❌ COLMAP SfM: FAILED - {e}")

if __name__ == "__main__":
    print("--- Independent Verification of AI Models and Tools ---")
    check_yolo()
    check_sam2()
    check_slam3r()
    check_colmap()
    print("-------------------------------------------------------")


import torch
import cv2
import numpy as np
import os
import sys

# Add SAM2 to path if needed
sys.path.append(os.path.abspath('sam2'))

try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2

    print("SAM2 modules imported successfully.")
    
    # Initialize the SAM2 image predictor
    # We will use the tiny model for a quick check, assuming they have it, or use what is in checkpoints
    checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pt')]
    if not checkpoints:
        print("No SAM2 checkpoints found in checkpoints/")
        sys.exit(1)
        
    ckpt_path = os.path.join('checkpoints', checkpoints[0])
    # Infer config from checkpoint name
    if 'tiny' in ckpt_path:
        cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    elif 'small' in ckpt_path:
        cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    elif 'base' in ckpt_path or 'b+' in ckpt_path:
        cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif 'large' in ckpt_path:
        cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    else:
        cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
    print(f"Loading SAM2 model: {ckpt_path} with config {cfg}")
    
    sam2_model = build_sam2(cfg, ckpt_path, device='cuda')
    predictor = SAM2ImagePredictor(sam2_model)
    print("[OK] SAM2 model built and predictor initialized successfully!")
    
    # Create a dummy image
    dummy_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    cv2.circle(dummy_image, (512, 512), 100, (255, 255, 255), -1)
    
    predictor.set_image(dummy_image)
    
    # Point prompt at the center of the white circle
    input_point = np.array([[512, 512]])
    input_label = np.array([1])
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    
    print(f"[OK] SAM2 Inference successful! Generated mask of shape: {masks.shape} with confidence score: {scores[0]:.3f}")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[FAILED] SAM2 Inference FAILED: {e}")
