import argparse
import numpy as np
import cv2 as cv
from models.mp_palmdet import MPPalmDet

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

    # 11. Resize the ROI to 224x224
    roi_resized = cv.resize(roi, (224, 224), interpolation=cv.INTER_LINEAR)

    return roi_resized

def visualize(image, results, print_results=False, fps=None):
    output = image.copy()

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    for idx, palm in enumerate(results):
        score = palm[-1]
        palm_box = palm[0:4]
        palm_landmarks = palm[4:-1].reshape(7, 2)

        # Add score
        palm_box = palm_box.astype(np.int32)
        cv.putText(output, '{:.4f}'.format(score), (palm_box[0], palm_box[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

        # Draw bounding box
        cv.rectangle(output, (palm_box[0], palm_box[1]), (palm_box[2], palm_box[3]), (0, 255, 0), 2)

        # Draw key points
        palm_landmarks = palm_landmarks.astype(np.int32)
        for p in palm_landmarks:
            cv.circle(output, p, 2, (0, 0, 255), 2)

        # Extract and visualize palm ROI
        palm_roi = extract_palm_roi(image, palm_landmarks)
        if palm_roi is not None:
            cv.imshow(f'Palm ROI {idx + 1}', palm_roi)

        # Print results
        if print_results:
            print('-----------palm {}-----------'.format(idx + 1))
            print('score: {:.2f}'.format(score))
            print('palm box: {}'.format(palm_box))
            print('palm landmarks: ')
            for plm in palm_landmarks:
                print('\t{}'.format(plm))
    return output

if __name__ == '__main__':
    backend_id = cv.dnn.DNN_BACKEND_OPENCV
    target_id = cv.dnn.DNN_TARGET_CPU

    model_path = r'weights/palm_detection_mediapipe_2023feb.onnx'
    nms_threshold = 0.3
    score_threshold = 0.8

    # Instantiate MPPalmDet
    model = MPPalmDet(modelPath=model_path,
                      nmsThreshold=nms_threshold,
                      scoreThreshold=score_threshold,
                      backendId=backend_id,
                      targetId=target_id)

    deviceId = 0
    cap = cv.VideoCapture(deviceId)

    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        results = model.infer(frame)
        tm.stop()

        # Draw results on the input image
        frame = visualize(frame, results, fps=tm.getFPS())

        # Visualize results in a new window
        cv.imshow('MPPalmDet Demo', frame)

        tm.reset()
