"""
python run.py
"""

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import onnxruntime
import time

# -----------------------------------------------------------------------------
reduce_res_factor = 1
thickness = 5

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

def masks_generator(frame):
    predictor.set_image(frame)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    print(f"image_embedding.shape: {image_embedding.shape}")

    # The following inputs must all be supplied to the ONNX model. All inputs are np.float32.
    input_point = np.array([[frame.shape[1]//2, frame.shape[0]//2]])
    input_label = np.array([1])
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(onnx_coord, frame.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(frame.shape[:2], dtype=np.float32)
    }

    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold

    return masks

def apply_mask(image, mask, alpha):
    teal = np.array([217, 200, 123])
    h, w = mask.shape[-2:]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask] = teal
    masked_image = image * (1 - alpha * mask.reshape(h, w, 1)) + color_mask * alpha * mask.reshape(h, w, 1)
    return masked_image.astype(np.uint8)

def draw_contours(input_mask, thickness):
    # Convert the input boolean mask to uint8
    uint8_mask = (input_mask * 255).astype(np.uint8)
    
    # Create a contour mask
    contour_mask = np.zeros(uint8_mask.shape, dtype=np.uint8)

    # Find contours
    contours = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    # Draw contours on the contour mask
    for contour in contours:
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness)

    # Convert the contour mask back to a boolean type
    bool_contour_mask = contour_mask.astype(np.bool_)
    
    return bool_contour_mask


# Open the default camera (0 represents the default camera, change the index for other cameras)
cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()

    _, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"frame.shape: {frame.shape}")

    frame_resized = cv2.resize(frame, (frame.shape[1]//reduce_res_factor, frame.shape[0]//reduce_res_factor), interpolation = cv2.INTER_AREA)
    print(f"frame_resized.shape: {frame_resized.shape}")

    frame_np = np.array(frame_resized)
    print(f"frame_np.shape: {frame_np.shape}")

    masks = masks_generator(frame_resized)
    print(f"masks.shape: {masks.shape}")
    # print(f"masks.dtype: {masks.dtype}")
    print(f"masks any: {np.any(masks)}")

    mask = masks[0]
    print(f"mask.shape: {mask.shape}")

    mask_2d = np.squeeze(mask)
    print(f"mask_2d.shape: {mask_2d.shape}")
    merged_frame = apply_mask(frame_np, mask_2d, alpha=0.33)

    mask_outline = draw_contours(mask_2d, thickness=thickness)
    merged_frame = apply_mask(merged_frame, mask_outline, alpha=1)

    # classification label

    center = (frame_resized.shape[1]//2, frame_resized.shape[0]//2)
    radius = thickness * 2
    teal = (217, 200, 123)
    width = thickness
    cv2.circle(merged_frame, center, radius, teal, width)

    end_time = time.time()
    print(end_time - start_time)
    print()
    cv2.imshow('Eye of Segmento', merged_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()