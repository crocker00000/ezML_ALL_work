# NOTE:
# This file was not used by being run directly
# Rather, the API on RapidAPI.com uses this code for foreground segmentation
    # https://rapidapi.com/api4ai-api4ai-default/api/cars-image-background-removal

# Download googleapis img (original car) using powershell:
    # Invoke-WebRequest -Uri "https://storage.googleapis.com/api4ai-static/samples/img-bg-removal-cars-1.jpg" -OutFile "C:\Users\colin\Documents\JOBS\ezML_internship\Carviz_window_transparency\Extract_car_from_img\Extract_car_from_img.jpg"


from schemas.function_schemas import CarVizReplaceBackground
from integrations.rapidapi import RapidApi
import base64
from PIL import Image
import io

import numpy as np
import cv2

"""
Considerations

- Currently returns in 3200x1800 resolution
- Should car be upscaled/repaired too (like designify) or do you guys alr do that?
"""

def replace_background(req: CarVizReplaceBackground):
    """
    Calls API that removes background and adds shadow to car
    Then places it on a background
    """

    car = RapidApi.cars_image_background_removal(req)

    if car is None:
        return {"status": "error", "detail": f"failed processing car image"}

    image = place_car_on_background(np.array(car), 'assets/concrete-floor_2_upscaled.png')

    image = Image.fromarray(image).convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")

    # Base64 encode the BytesIO object
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"status": "success", "result": img_str}

def place_car_on_background(car_image: np.array, background_image_path: str):
    """
    First scales the car_image, then places it on the background_image
    """
    background_image = cv2.imread(background_image_path)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

    # Find the non-transparent region (where alpha is not zero)
    alpha_channel = car_image[:, :, 3]
    ys, xs = np.nonzero(alpha_channel)

    # Check if the car is found in the image
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No non-transparent pixels found in the car image.")

    # Find bounding box of the car
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Crop the car image to the bounding box
    car_image = car_image[y_min:y_max+1, x_min:x_max+1]

    # Desired size for the car (e.g., 50% of background's width and height)
    desired_car_width = int(background_image.shape[1] * 0.7) # 0.7
    desired_car_height = int(background_image.shape[0] * 0.75) # 0.9

    # Calculate scale factors for both width and height
    scale_factor_width = desired_car_width / car_image.shape[1]
    scale_factor_height = desired_car_height / car_image.shape[0]

    # Choose the smaller scale factor to ensure the car image does not exceed 50% of either dimension
    scale_factor = min(scale_factor_width, scale_factor_height)
    
    # Make sure the scaled car height does not exceed background height
    if car_image.shape[0] * scale_factor > background_image.shape[0]:
        scale_factor = background_image.shape[0] / car_image.shape[0]

    # Resize car image
    new_size = (int(car_image.shape[1] * scale_factor), int(car_image.shape[0] * scale_factor))
    car_image = cv2.resize(car_image, new_size, interpolation=cv2.INTER_AREA)

    return center_overlay(background_image, car_image)

def center_overlay(bg_image, overlay_image):
    """
    Places overlay_image on top of bg_image at the center
    """
    CENTER_X_OFFSET = 1
    CENTER_Y_OFFSET = 1.5
    # Calculate the center position
    center_x = int((bg_image.shape[1] - overlay_image.shape[1]) // 2 * CENTER_X_OFFSET)
    center_y = int((bg_image.shape[0] - overlay_image.shape[0]) // 2 * CENTER_Y_OFFSET)

    center_x = max(0, min(center_x, bg_image.shape[1] - overlay_image.shape[1]))
    center_y = max(0, min(center_y, bg_image.shape[0] - overlay_image.shape[0]))

    # Overlay the image
    y1, y2 = center_y, center_y + overlay_image.shape[0]
    x1, x2 = center_x, center_x + overlay_image.shape[1]

    alpha_overlay = overlay_image[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        bg_image[y1:y2, x1:x2, c] = (alpha_overlay * overlay_image[:, :, c] +
                                     alpha_background * bg_image[y1:y2, x1:x2, c])

    return bg_image

