import argparse

import numpy as np
import cv2 as cv
from models.mp_palmdet import MPPalmDet

def visualize(image, results, print_results=False, fps=None):
    output = image.copy()

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    for idx, palm in enumerate(results):
        score = palm[-1]
        palm_box = palm[0:4]
        palm_landmarks = palm[4:-1].reshape(7, 2)

        # put score
        palm_box = palm_box.astype(np.int32)
        cv.putText(output, '{:.4f}'.format(score), (palm_box[0], palm_box[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

        # draw box
        cv.rectangle(output, (palm_box[0], palm_box[1]), (palm_box[2], palm_box[3]), (0, 255, 0), 2)

        # draw points
        palm_landmarks = palm_landmarks.astype(np.int32)
        for p in palm_landmarks:
            cv.circle(output, p, 2, (0, 0, 255), 2)

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

    model_path = r'assets/palm_detection_mediapipe_2023feb.onnx'
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

        # Visualize results in a new Window
        cv.imshow('MPPalmDet Demo', frame)

        tm.reset()
