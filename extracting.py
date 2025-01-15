import cv2 as cv
import numpy as np
from models.mobileNetV1 import MobileNetV1
from recognizing import recognize_palm

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_path = r'assets/model_best_mobilenetV1.pth'


def extract_palm_roi(image, palm_landmarks):
    # 1. Convert to integer coordinates
    palm_landmarks = palm_landmarks.astype(np.int32)

    # 2. Perform linear fitting using 4 points (excluding the 0th point - bottom_point)
    x_coords, y_coords = palm_landmarks[1:5, 0], palm_landmarks[1:5, 1]
    slope, intercept = np.polyfit(x_coords, y_coords, 1)

    # 3. Calculate the rotation angle based on the slope
    angle = np.degrees(np.arctan(slope))

    # 4. Rotate the image with the center of the image as reference
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale=1.0)
    rotated_image = cv.warpAffine(image, rotation_matrix, (w, h))

    # 5. Update all key points to rotated coordinates
    ones = np.ones((palm_landmarks.shape[0], 1), dtype=np.int32)
    rotated_landmarks = (rotation_matrix @ np.hstack([palm_landmarks, ones]).T).T.astype(np.int32)

    # 6. Perform linear fitting again in the rotated coordinate system using 4 points
    rx_coords, ry_coords = rotated_landmarks[1:5, 0], rotated_landmarks[1:5, 1]
    rslope, rintercept = np.polyfit(rx_coords, ry_coords, 1)

    # 7. Calculate the vertical distance from the bottom point (rotated_landmarks[0]) to the fitted line
    x_bottom, y_bottom = rotated_landmarks[0]
    y_line_bottom = int(rslope * x_bottom + rintercept)
    distance = abs(y_bottom - y_line_bottom)

    # 8. Determine the boundaries of the ROI based on this distance
    x_top_start = x_bottom - distance // 2
    x_top_end   = x_bottom + distance // 2
    y_top_start = int(rslope * x_top_start + rintercept)
    y_top_end   = int(rslope * x_top_end   + rintercept)

    # 9. Ensure the ROI is within the image range
    x1, x2 = max(0, x_top_start), min(w, x_top_end)
    y1, y2 = max(0, min(y_top_start, y_bottom - distance)), min(h, max(y_top_end, y_bottom))

    # 10. Crop the ROI
    roi = rotated_image[y1:y2, x1:x2]

    # 11. Resize the ROI to 128x128
    roi_resized = cv.resize(roi, (128, 128), interpolation=cv.INTER_LINEAR)

    predicted_label = recognize_palm(roi_resized, model_path, num_classes=293)

    return roi_resized
