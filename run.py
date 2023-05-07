"""
python run.py
"""

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import onnxruntime
import matplotlib.pyplot as plt
import time

# -----------------------------------------------------------------------------
webcam_w = 1920
webcam_h = 1080
reduce_res_factor = 2

checkpoint = "./models/sam_vit_b_01ec64.pth"
model_type = "vit_b"
onnx_model_path = "./onnx/sam_onnx_b.onnx"
onnx_model_quantized_path = "./onnx/sam_onnx_b_quantized.onnx"
# onnx_model_path = onnx_model_quantized_path

# To use the ONNX model, the image must first be pre-processed using the SAM image encoder.
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='mps')
predictor = SamPredictor(sam)

ort_session = onnxruntime.InferenceSession(onnx_model_path)
# -----------------------------------------------------------------------------






import numpy as np

def turn_black(np_frame, mask_2d):
    # Check if shapes match
    if np_frame.shape[0] != mask_2d.shape[0] or np_frame.shape[1] != mask_2d.shape[1]:
        raise ValueError("np_frame and mask_2d shapes do not match")

    # Create a new NumPy array with the same shape as np_frame
    blackened_frame = np_frame.copy()

    # Use advanced indexing to turn the specified pixels black
    blackened_frame[mask_2d] = 0

    return blackened_frame







def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   




# Open the default camera (0 represents the default camera, change the index for other cameras)
cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()

    _, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"frame.shape: {frame.shape}")

    frame_resized = cv2.resize(frame, (frame.shape[1]//reduce_res_factor, frame.shape[0]//reduce_res_factor), interpolation = cv2.INTER_AREA)
    print(f"frame_resized.shape: {frame_resized.shape}")

    np_frame = np.array(frame_resized)
    print(f"np_frame.shape: {np_frame.shape}")

    predictor.set_image(frame_resized)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    print(f"image_embedding.shape: {image_embedding.shape}")

    # The following inputs must all be supplied to the ONNX model. All inputs are np.float32.
    input_point = np.array([[frame_resized.shape[1]//2, frame_resized.shape[0]//2]])
    input_label = np.array([1])
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(onnx_coord, frame_resized.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(frame_resized.shape[:2], dtype=np.float32)
    }

    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold
    print(f"masks.shape: {masks.shape}")
    print(f"masks.dtype: {masks.dtype}")
    print(f"masks any: {np.any(masks)}")
    mask = masks[0]
    print(f"mask.shape: {mask.shape}")
    mask_2d = np.squeeze(mask)
    print(f"mask_2d.shape: {mask_2d.shape}")
    # Create a 3-channel bool array image with the same shape as np_frame
    mask_3c = np.stack((mask_2d, mask_2d, mask_2d), axis=-1)
    print(f"mask_3c.shape: {mask_3c.shape}")

    merged_frame = turn_black(np_frame, mask_2d)
    # edge
    # box


    center = (frame_resized.shape[1]//2, frame_resized.shape[0]//2)
    radius = 4
    white = (255, 255, 255)
    width = 2
    cv2.circle(merged_frame, center, radius, white, width)

    end_time = time.time()
    print(end_time - start_time)
    print()
    cv2.imshow('Webcam Input', merged_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()