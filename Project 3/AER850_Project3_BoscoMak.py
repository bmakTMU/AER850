"""
AER850 - Project 3
PCB Masking + YOLOv11 Training + Evaluation

- Step 1: Mask PCB from motherboard_image.JPEG using OpenCV
- Step 2: Train YOLOv11 (nano) model on provided PCB dataset (Ultralytics)
- Step 3: Evaluate trained model on 3 images in Evaluation folder

Run this script from the project root (or adjust paths below).
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path

# If you don't have ultralytics installed, run:
#   pip install ultralytics
from ultralytics import YOLO


# =========================
#  CONFIGURATION
# =========================

# ---- Step 1: Masking paths ----
MOTHERBOARD_IMAGE_PATH = "Project 3 Data/motherboard_image.JPEG"   # update if needed
MASK_OUTPUT_DIR = "outputs/masking"

# ---- Step 2: YOLO training configuration ----
# Path to data YAML file for YOLO (provided with dataset)
YOLO_DATA_YAML = "Project 3 Data/data/data.yaml"       # update to your actual path
YOLO_PRETRAINED_WEIGHTS = "yolo11n.pt"              # YOLOv11 nano weights (put in working dir or give full path)
YOLO_RUN_NAME = "pcb_yolo11n"                       # folder name under runs/detect/

EPOCHS = 100        # stay below 200 as required
BATCH_SIZE = 16     # adjust based on GPU memory
IMG_SIZE = 1024     # minimum recommended is 900


# ---- Step 3: Evaluation configuration ----
EVAL_IMAGES_DIR = "Project 3 Data/data/test"     # folder with 3 test images
EVAL_OUTPUT_DIR = "outputs/evaluation"              # where predictions will be saved
CONF_THRESHOLD = 0.25                               # confidence threshold for detection


# =========================
#  STEP 1: OBJECT MASKING
# =========================

def mask_pcb(
    image_path: str,
    output_dir: str,
    blur_ksize: int = 5,
    canny_thresh1: int = 50,
    canny_thresh2: int = 150,
):
    """
    Load motherboard image, detect PCB contour, create mask, and extract PCB.

    Saves:
      - gray image
      - edge image (Canny)
      - binary mask
      - extracted PCB

    Returns:
      extracted_path: path to saved extracted PCB image
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Read the RGB image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    # 2) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3) Blur to reduce noise before edge detection
    gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # 4) Edge detection (Canny)
    edges = cv2.Canny(gray_blur, canny_thresh1, canny_thresh2)

    # 5) Find contours on the edge map
    # RETR_EXTERNAL: only outermost contours (we expect PCB to be one of them)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contours found in image – check threshold / Canny parameters.")

    # 6) Choose the largest contour by area – likely the PCB
    largest_contour = max(contours, key=cv2.contourArea)

    # Optional: filter out very small contours if needed
    contour_area = cv2.contourArea(largest_contour)
    h, w = gray.shape
    img_area = h * w

    # If the largest contour is suspiciously small, warn the user
    if contour_area < 0.05 * img_area:
        print("Warning: Largest contour area is small relative to image. "
              "You may need to tune threshold / Canny parameters.")

    # 7) Create an empty mask and draw the PCB contour filled in white
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], contourIdx=-1, color=255, thickness=cv2.FILLED)

    # 8) Use the mask to extract the PCB region
    # bitwise_and keeps pixels where mask is non-zero
    extracted = cv2.bitwise_and(img, img, mask=mask)

    # 9) Save intermediate results for report
    gray_path       = os.path.join(output_dir, "motherboard_gray.png")
    edges_path      = os.path.join(output_dir, "motherboard_edges.png")
    mask_path       = os.path.join(output_dir, "motherboard_mask.png")
    extracted_path  = os.path.join(output_dir, "motherboard_extracted.png")

    cv2.imwrite(gray_path, gray)
    cv2.imwrite(edges_path, edges)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(extracted_path, extracted)

    print(f"[Masking] Saved gray image to:      {gray_path}")
    print(f"[Masking] Saved edge image to:      {edges_path}")
    print(f"[Masking] Saved mask image to:      {mask_path}")
    print(f"[Masking] Saved extracted PCB to:   {extracted_path}")

    return extracted_path


# =========================
#  STEP 2: YOLOv11 TRAINING
# =========================

def train_yolo(
    data_yaml: str,
    pretrained_weights: str,
    run_name: str,
    epochs: int,
    batch: int,
    imgsz: int,
):
    """
    Train a YOLOv11 model (nano) using Ultralytics.

    Args:
        data_yaml: path to YAML config describing train/val/test sets & classes
        pretrained_weights: path or name of YOLOv11 weights (e.g., 'yolo11n.pt')
        run_name: name of the run folder under runs/detect/
        epochs: number of epochs (must be < 200)
        batch: batch size
        imgsz: image size (single int, min recommended ~900)
    """
    # Load model from pretrained weights
    print("[YOLO] Loading model...")
    model = YOLO(pretrained_weights)

    print("[YOLO] Starting training...")
    # model.train() automatically creates runs/detect/<run_name> folder
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name=run_name,
    )

    print("[YOLO] Training complete.")
    print(f"[YOLO] Run folder: runs/detect/{run_name}")
    return results


# =========================
#  STEP 3: EVALUATION
# =========================

def evaluate_yolo_on_folder(
    run_name: str,
    eval_dir: str,
    output_dir: str,
    conf: float = 0.25,
):
    """
    Use the best model from a YOLO run to predict on all images in eval_dir.

    Saves:
      - predicted images with bounding boxes into output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    # Path to best weights from training (Ultralytics standard)
    best_weights = f"runs/detect/{run_name}/weights/best.pt"

    if not os.path.exists(best_weights):
        raise FileNotFoundError(f"Could not find trained weights at: {best_weights}")

    # Load trained model
    print("[Eval] Loading trained model...")
    model = YOLO(best_weights)

    # Collect evaluation images (jpg/jpeg/png)
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    eval_paths = []
    for ext in image_extensions:
        eval_paths.extend(glob.glob(os.path.join(eval_dir, ext)))

    if not eval_paths:
        raise FileNotFoundError(f"No images found in evaluation directory: {eval_dir}")

    print(f"[Eval] Found {len(eval_paths)} evaluation image(s).")

    for img_path in eval_paths:
        print(f"[Eval] Predicting on {img_path} ...")
        # predict() can automatically save images with drawn boxes
        # We specify project/output to keep things tidy.
        model.predict(
            source=img_path,
            conf=conf,
            save=True,
            project=output_dir,
            name="pcb_eval",  # predictions will go in output_dir/pcb_eval/
            exist_ok=True,    # do not create new folder for repeated runs
        )

    print(f"[Eval] Predictions saved under: {output_dir}/pcb_eval")


# =========================
#  MAIN
# =========================

def main():
    # -------- Step 1: Masking --------
    print("========== STEP 1: MASK PCB ==========")
    extracted_pcb_path = mask_pcb(
        image_path=MOTHERBOARD_IMAGE_PATH,
        output_dir=MASK_OUTPUT_DIR,
    )

    print(f"[Main] Extracted PCB image saved at: {extracted_pcb_path}")
    print("Use the saved images (gray, edges, mask, extracted) in your report.")

    # -------- Step 2: YOLO Training --------
    print("\n========== STEP 2: TRAIN YOLOv11 ==========")
    print("This step may take a long time depending on your GPU.")
    print("Make sure YOLO_DATA_YAML and YOLO_PRETRAINED_WEIGHTS paths are correct.")

    # Comment this out if you want to run training separately (e.g., in Colab).
    train_yolo(
        data_yaml=YOLO_DATA_YAML,
        pretrained_weights=YOLO_PRETRAINED_WEIGHTS,
        run_name=YOLO_RUN_NAME,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
    )

    # -------- Step 3: Evaluation --------
    print("\n========== STEP 3: EVALUATION ==========")
    evaluate_yolo_on_folder(
        run_name=YOLO_RUN_NAME,
        eval_dir=EVAL_IMAGES_DIR,
        output_dir=EVAL_OUTPUT_DIR,
        conf=CONF_THRESHOLD,
    )

    print("\nAll steps completed.")
    print("Remember to:")
    print(" - Include masking figures (edges, mask, extracted PCB) in your report.")
    print(" - Include YOLO training plots from runs/detect/<run_name>/")
    print(" - Show evaluation images and discuss missed/mislabeled components.")


if __name__ == "__main__":
    main()
