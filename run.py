"""
python run.py
"""

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import onnxruntime
import time
import math
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms as T
from models import *
from datasets import ImageNet

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
classification_model_name = "ConvNeXt"
classification_model_variant = "T"
classification_model_checkpoint = "./models/convnext_tiny_1k_224_ema.pth"
classification_image_size = 224
# -----------------------------------------------------------------------------

def generate_mask(frame):
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
    print(f"image_embedding.shape: {image_embedding.shape}")

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

def overlay_mask(image, mask, alpha):
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

def calculate_bbox_coords(image_np, text, font_size, position, bbox_padding, outline_thickness):
    """
    Calculates the (top, bottom, left, right) coordinates of the text bbox rectangle given the input parameters.

    Args:
    image_np (numpy.ndarray): The input image.
    text (str): The text to add.
    font_size (int): The font size.
    position (tuple): The position to add the text.
    bbox_padding (int): The padding between the text and the bbox.
    outline_thickness (int): The thickness of the outline around the bbox.

    Returns:
    tuple: The (top, bottom, left, right) coordinates of the text bbox rectangle.
    """
    # Convert the NumPy array back to a PIL image
    image_pil = Image.fromarray(image_np)

    # Define the font type and size
    font_path = hud_font
    font = ImageFont.truetype(font_path, font_size)

    # Create a drawing context
    draw = ImageDraw.Draw(image_pil)

    # Calculate the text size using the textbbox() function
    text_bbox = draw.textbbox(position, text, font=font)

    # Define the rectangle coordinates
    top = text_bbox[1] - bbox_padding
    bottom = text_bbox[3] + bbox_padding
    left = text_bbox[0] - bbox_padding
    right = text_bbox[2] + bbox_padding

    return (top, bottom, left, right)

def add_highlighted_text(image_np, text, font_size, font_color, position, bbox_color, bbox_padding, outline_color, outline_thickness, alpha):
    """
    Adds text to the input image at the specified position using the specified font and draws a box with an outline around it.
    Args:
    image_np (numpy.ndarray): The input image.
    text (str): The text to add.
    font_size (int): The font size.
    font_color (tuple): The font color.
    position (tuple): The position to add the text.
    box_color (tuple): The color of the box around the text.
    box_padding (int): The padding between the text and the box.
    outline_color (tuple): The color of the outline around the box.
    outline_thickness (int): The thickness of the outline around the box.
    alpha (float): The transparency level of the box fill (0 to 1).
    Returns:
    numpy.ndarray: The modified image with the added text and box with an outline.
    """
    # Convert the NumPy array back to a PIL image
    image_pil = Image.fromarray(image_np)

    # Define the font type, size, and color
    font_path = hud_font
    font = ImageFont.truetype(font_path, font_size)

    # Create a drawing context
    draw = ImageDraw.Draw(image_pil, "RGBA")

    # Calculate the text size using the textbbox() function
    text_bbox = draw.textbbox(position, text, font=font)

    # Draw a rectangle with a transparent background around the text
    draw.rectangle(
        [text_bbox[0] - bbox_padding, text_bbox[1] - bbox_padding, text_bbox[2] + bbox_padding, text_bbox[3] + bbox_padding],
        fill=bbox_color + (int(255 * alpha),),  # RGBA: Add an alpha channel to the box color for transparency
        outline=outline_color,
        width=outline_thickness
    )

    # Add the text to the image using the text() function with a transparent background
    draw.text(position, text, font=font, fill=font_color, stroke_width=0)

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

def clear_center(bool_array, hud_thickness=hud_thickness):
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

class ModelInference:
    def __init__(self, model: str, variant: str, checkpoint: str, size: int) -> None:
        """
        Initialize the ModelInference class.

        Args:
        model (str): The model class name.
        variant (str): The model variant name.
        checkpoint (str): The model checkpoint file.
        size (int): The input image size for the model.
        """
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Dataset class labels
        self.labels = ImageNet.CLASSES
        
        # Initialize the model with the provided variant, checkpoint, and number of class labels
        self.model = eval(model)(variant, checkpoint, len(self.labels), size)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define the preprocessing pipeline
        self.preprocess = T.Compose([
            # Normalize pixel values to [0, 1]
            T.Lambda(lambda x: x / 255),
            # Resize the input image to the specified size
            T.Resize((size, size)),
            # Normalize with ImageNet mean and std
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # Add an additional batch dimension
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def __call__(self, classification_input_tensor) -> str:
        """
        Perform inference on the provided input tensor.

        Args:
        classification_input_tensor (torch.Tensor): The input tensor for classification.

        Returns:
        str: The predicted class label.
        """
        image = classification_input_tensor
        
        # Preprocess the input tensor
        image = self.preprocess(image).to(self.device)
        
        # Perform model inference
        with torch.inference_mode():
            pred = self.model(image)
        
        # Postprocess the output to get the class label
        cls_name = self.labels[pred.argmax()]
        return cls_name

def overlay_gray_mask(image, mask):
    """
    Applies a gray mask on the input image wherever the mask is False.

    Args:
    image (numpy.ndarray): The input image.
    mask (numpy.ndarray): The binary mask to apply.

    Returns:
    numpy.ndarray: The masked image.
    """
    h, w = mask.shape[-2:]
    gray_mask = np.full((h, w, 3), 128, dtype=np.uint8)
    masked_image = np.where(mask.reshape(h, w, 1), image, gray_mask)
    return masked_image.astype(np.uint8)

def crop_image_by_mask(image, mask):
    """
    Crops the input image based on the bbox defined by the True values in the mask array.

    Args:
    image (numpy.ndarray): The input image.
    mask (numpy.ndarray): The 2D bool array.

    Returns:
    numpy.ndarray: The cropped image.
    """
    # Find the indices of the True values in the mask
    true_indices = np.argwhere(mask)

    # Get the top, bottom, left, and right coordinates of the bbox
    top = true_indices[:, 0].min()
    bottom = true_indices[:, 0].max()
    left = true_indices[:, 1].min()
    right = true_indices[:, 1].max()

    # Crop the image using the bbox coordinates
    cropped_image = image[top:bottom+1, left:right+1]

    return cropped_image

def clear_textbbox(mask_2d, textbbox_coords, set_value):
    """
    Modifies the input 2D boolean numpy array mask_2d, setting the specified boolean value for any pixels within the input text bbox.

    Args:
    mask_2d (numpy.ndarray): A 2D boolean numpy array representing the mask.
    textbbox_coords (tuple): A tuple containing the (top, bottom, left, right) coordinates of the text bbox.
    set_value (bool): The boolean value to set for the pixels within the text bbox.

    Returns:
    numpy.ndarray: The modified 2D boolean numpy array with the specified boolean value for any pixels within the input text bbox.
    """
    top, bottom, left, right = [int(coord) for coord in textbbox_coords]

    # Check if the coordinates are within the mask boundaries
    mask_height, mask_width = mask_2d.shape
    top = max(0, top)
    bottom = min(mask_height, bottom)
    left = max(0, left)
    right = min(mask_width, right)

    # Set the values within the text bbox to the specified value
    mask_2d[top:bottom, left:right] = set_value

    return mask_2d

# -----------------------------------------------------------------------------
# Model setup
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='mps' if torch.backends.mps.is_available() else 'cpu')
predictor = SamPredictor(sam)
ort_session = onnxruntime.InferenceSession(onnx_model_path)
classification_model = ModelInference(model=classification_model_name, 
                                      variant=classification_model_variant, 
                                      checkpoint=classification_model_checkpoint, 
                                      size=classification_image_size)
# -----------------------------------------------------------------------------

# Open the default camera (0 represents the default camera, change the index for other cameras)
cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    print(f"mps check: {'mps' if torch.backends.mps.is_available() else 'cpu'}")

    # Capture a single frame
    _, frame = cap.read()
    print(f"frame.shape: {frame.shape}")

    # Resize the frame
    frame_resized = cv2.resize(frame, (frame.shape[1]//frame_resize_factor, frame.shape[0]//frame_resize_factor), interpolation=cv2.INTER_AREA)
    print(f"frame_resized.shape: {frame_resized.shape}")

    # Convert the resized frame to a NumPy array
    frame_np = np.array(frame_resized)
    print(f"frame_np.shape: {frame_np.shape}")

    # Generate masks
    masks = generate_mask(frame_resized)
    print(f"masks.shape: {masks.shape}")
    print(f"masks.dtype: {masks.dtype}")
    print(f"np.any(masks): {np.any(masks)}")

    # Extract the first mask
    mask = masks[0]
    print(f"mask.shape: {mask.shape}")
    mask_2d = np.squeeze(mask)
    print(f"mask_2d.shape: {mask_2d.shape}")

    # Image classification
    classification_input_np_overlay = overlay_gray_mask(frame_np, mask_2d)
    classification_input_np_cropped = crop_image_by_mask(classification_input_np_overlay, mask_2d)
    classification_input_tensor = torch.from_numpy(classification_input_np_cropped)
    classification_input_tensor = classification_input_tensor.permute(2, 0, 1)
    print(f"classification_input_tensor.shape: {classification_input_tensor.shape}")
    cls_name = classification_model(classification_input_tensor)
    print(f"classification result: {cls_name.capitalize()}")

    # Darken the frame
    frame_np = darken_image(frame_np, alpha=0.5)

    # Values for later
    frame_x, frame_y = frame_np.shape[1], frame_np.shape[0]
    center_circle_radius = int(math.sqrt((hud_thickness**2)*2))

    # mask_area
    mask_area = np.copy(mask_2d)
    textbbox_coords = calculate_bbox_coords(frame_np, text=cls_name.capitalize(), font_size=32, bbox_padding=int(hud_thickness*2), outline_thickness=int(hud_thickness*1.5), position=((frame_x//2)+(frame_y//8)+(frame_y//8)+hud_thickness*1.5, (frame_y//2)-(frame_y//8)-hud_thickness*3))
    mask_area = clear_textbbox(mask_area, textbbox_coords, set_value=True)
    frame_np = overlay_mask(frame_np, mask_area, alpha=0.33)

    # mask_outline
    mask_outline = np.copy(mask_2d)
    mask_outline = draw_contours(mask_outline, thickness=hud_thickness)
    mask_outline = clear_textbbox(mask_outline, textbbox_coords, set_value=False)
    mask_outline = clear_center(mask_outline)
    frame_np = overlay_mask(frame_np, mask_outline, alpha=1)

    # Draw the center circle
    center_coords = (frame_np.shape[1]//2, frame_np.shape[0]//2)
    circle_radius = hud_thickness * 2
    circle_width = hud_thickness
    cv2.circle(frame_np, center_coords, circle_radius, hud_color, circle_width)
    # Draw the lines
    cv2.line(frame_np, ((frame_x//2)+center_circle_radius, (frame_y//2)-center_circle_radius), ((frame_x//2)+(frame_y//8), (frame_y//2)-(frame_y//8)), hud_color, hud_thickness)
    cv2.line(frame_np, ((frame_x//2)+(frame_y//8), (frame_y//2)-(frame_y//8)), ((frame_x//2)+(frame_y//8)+(frame_y//8), (frame_y//2)-(frame_y//8)), hud_color, hud_thickness)
    frame_np = add_highlighted_text(frame_np, text=cls_name.capitalize(), font_size=32, font_color=hud_color, bbox_color=hud_color, bbox_padding=int(hud_thickness*2), outline_color=hud_color, alpha=0, outline_thickness=int(hud_thickness*1.5), position=((frame_x//2)+(frame_y//8)+(frame_y//8)+hud_thickness*1.5, (frame_y//2)-(frame_y//8)-hud_thickness*3))
    # Add diagnostic infos
    frame_np = add_text(frame_np, text=f"frame_np.shape: {frame_np.shape}", font_size=28, font_color=hud_color, position=(30, frame_np.shape[0]-180))
    frame_np = add_text(frame_np, text=f"mask_2d.shape: {mask_2d.shape}", font_size=28, font_color=hud_color, position=(30, frame_np.shape[0]-140))
    frame_np = add_text(frame_np, text=f"np.any(mask_2d): {np.any(mask_2d)}", font_size=28, font_color=hud_color, position=(30, frame_np.shape[0]-100))
    frame_np = add_text(frame_np, text=f"classification_input_tensor.shape: {classification_input_tensor.shape}", font_size=28, font_color=hud_color, position=(30, frame_np.shape[0]-60))
    # Add FPS
    end_time = time.time()
    frame_np = add_text(frame_np, text=f"FPS: {1/(end_time-start_time):.2f}", font_size=28, font_color=hud_color, position=(30, 30))
    print(f"{end_time-start_time:.2f}")
    print("-----------------------------------------------------------------------------")

    cv2.imshow('Eye of Segmento', frame_np)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
