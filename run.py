"""
python run.py
"""

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import onnxruntime
import time
from PIL import Image, ImageDraw, ImageFont

# -----------------------------------------------------------------------------
# Configuration
frame_resize_factor = 1
hud_thickness = 5
hud_color = (217, 200, 123)
hud_font = '/Users/star-bits/Library/Fonts/FontsFree-Net-SFMono-Bold.ttf'
checkpoint = "./models/sam_vit_b_01ec64.pth"
model_type = "vit_b"
onnx_model_path = "./onnx/sam_onnx_b.onnx"
onnx_model_quantized_path = "./onnx/sam_onnx_b_quantized.onnx"
# Model setup
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='mps')
predictor = SamPredictor(sam)
ort_session = onnxruntime.InferenceSession(onnx_model_path)
# -----------------------------------------------------------------------------

def masks_generator(frame):
    """
    Generates segmentation masks using the SAM predictor and ONNX model.
    Args:
    frame (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The generated masks.
    """
    # Get image embedding from SAM predictor
    predictor.set_image(frame)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    # print(f"image_embedding.shape: {image_embedding.shape}")

    # The following inputs must all be supplied to the ONNX model. All inputs are np.float32.
    input_point = np.array([[frame.shape[1]//2, frame.shape[0]//2]])
    input_label = np.array([1])
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(onnx_coord, frame.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    # Build input dictionary for ONNX model
    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(frame.shape[:2], dtype=np.float32)
    }

    # Run ONNX model and get masks
    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold

    return masks

def apply_mask(image, mask, alpha):
    """
    Applies a colored mask on the input image with the given transparency.
    Args:
    image (numpy.ndarray): The input image.
    mask (numpy.ndarray): The binary mask to apply.
    alpha (float): The transparency level of the mask (0 to 1).

    Returns:
    numpy.ndarray: The masked image.
    """
    color = np.array(hud_color)
    h, w = mask.shape[-2:]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask] = color
    masked_image = image * (1 - alpha * mask.reshape(h, w, 1)) + color_mask * alpha * mask.reshape(h, w, 1)
    return masked_image.astype(np.uint8)

def draw_contours(input_mask, thickness):
    """
    Draws contours around the input mask.
    Args:
    input_mask (numpy.ndarray): The input binary mask.
    thickness (int): The thickness of the contours.

    Returns:
    numpy.ndarray: The binary mask with contours.
    """
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

def add_text(image_np, text, font_size, font_color, position):
    """
    Adds text to the input image at the specified position using the specified font.
    Args:
    image_np (numpy.ndarray): The input image.
    text (str): The text to add.
    font_size (int): The font size.
    font_color (tuple): The font color.
    position (tuple): The position to add the text.

    Returns:
    numpy.ndarray: The modified image with the added text.
    """
    # Convert the NumPy array back to a PIL image
    image_pil = Image.fromarray(image_np)

    # Define the font type, size, and color
    font_path = hud_font
    font = ImageFont.truetype(font_path, font_size)

    # Create a drawing context
    draw = ImageDraw.Draw(image_pil)

    # Add the text to the image using the text() function
    draw.text(position, text, font=font, fill=font_color)

    # Convert the modified PIL image back to a NumPy array
    image_text_np = np.array(image_pil)

    return image_text_np

def darken_image(image, alpha):
    """
    Darkens the input image by blending it with a black image using the specified alpha value.
    Args:
    image (numpy.ndarray): The input image.
    alpha (float): The blending factor (0 to 1).

    Returns:
    numpy.ndarray: The darkened image.
    """
    # Create a black image with the same shape as the input image
    black_image = np.zeros_like(image)

    # Blend the input image with the black image using the specified alpha value
    darkened_image = cv2.addWeighted(image, 1 - alpha, black_image, alpha, 0)

    return darkened_image

def clear_center_circle(bool_array, hud_thickness=hud_thickness):
    """
    Sets a circle with a radius of (hud_thickness * 2) in the center of the input 2D boolean array to False.

    Args:
    bool_array (numpy.ndarray): The input 2D boolean array.
    hud_thickness (int): The circle's radius is equal to hud_thickness * 2.

    Returns:
    numpy.ndarray: The modified 2D boolean array with the specified circle set to False.
    """
    h, w = bool_array.shape
    center_y, center_x = h // 2, w // 2
    radius = hud_thickness * 2

    # Create a meshgrid for the array indices
    y, x = np.ogrid[-center_y:h - center_y, -center_x:w - center_x]

    # Calculate the distance from the center for each point in the meshgrid
    distance_from_center = x**2 + y**2

    # Set points inside the circle to False
    bool_array[distance_from_center <= radius**2] = False

    return bool_array


# Open the default camera (0 represents the default camera, change the index for other cameras)
cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()

    # Capture the frame
    _, frame = cap.read()
    print(f"frame.shape: {frame.shape}")

    # Resize the frame
    frame_resized = cv2.resize(frame, (frame.shape[1]//frame_resize_factor, frame.shape[0]//frame_resize_factor), interpolation=cv2.INTER_AREA)
    print(f"frame_resized.shape: {frame_resized.shape}")

    # Convert the frame to a NumPy array
    frame_np = np.array(frame_resized)
    print(f"frame_np.shape: {frame_np.shape}")

    # Darken the frame
    frame_np = darken_image(frame_np, alpha=0.5)

    # Generate masks
    masks = masks_generator(frame_resized)
    print(f"masks.shape: {masks.shape}")
    print(f"masks.dtype: {masks.dtype}")
    print(f"masks any: {np.any(masks)}")

    # Get the first mask and apply it to the frame
    mask = masks[0]
    print(f"mask.shape: {mask.shape}")
    mask_2d = np.squeeze(mask)
    print(f"mask_2d.shape: {mask_2d.shape}")
    frame_np = apply_mask(frame_np, mask_2d, alpha=0.33)

    # Draw the mask contour
    mask_outline = draw_contours(mask_2d, thickness=hud_thickness)
    mask_outline = clear_center_circle(mask_outline)
    frame_np = apply_mask(frame_np, mask_outline, alpha=1)

    # Draw the center circle
    center = (frame_np.shape[1]//2, frame_np.shape[0]//2)
    radius = hud_thickness * 2
    width = hud_thickness
    cv2.circle(frame_np, center, radius, hud_color, width)

    # Classification

    # Calculate FPS and display it on the frame
    end_time = time.time()
    print(end_time - start_time)
    print("-----------------------------------------------------------------------------")
    frame_np = add_text(frame_np, text=f"FPS: {1/(end_time-start_time):.2f}", font_size=28, font_color=hud_color, position=(30, 30))

    # Display additional information on the frame
    frame_np = add_text(frame_np, text=f"frame_np.shape: {frame_np.shape}", font_size=28, font_color=hud_color, position=(30, frame_np.shape[0]-140))
    frame_np = add_text(frame_np, text=f"mask_2d.shape: {mask_2d.shape}", font_size=28, font_color=hud_color, position=(30, frame_np.shape[0]-100))
    frame_np = add_text(frame_np, text=f"mask_2d any: {np.any(mask_2d)}", font_size=28, font_color=hud_color, position=(30, frame_np.shape[0]-60))

    # Show the modified frame
    cv2.imshow('Eye of Segmento', frame_np)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
