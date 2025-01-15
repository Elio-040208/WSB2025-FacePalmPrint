import cv2 as cv
from models.mp_palmdet import MPPalmDet
from visualizing import visualize_hand_detection, visualize_counting
from recognizing import recognize_face_landmarks
from counting import update_counts

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
        frame = visualize_hand_detection(frame, results, fps=tm.getFPS())
        
        
        results = recognize_face_landmarks(frame)

        for ear, mar, face_distance, thresholds, counters in results:
            count_eye, total_blinks, count_mouth, total_mouth, distance_left, distance_right, total_face = update_counts(
                ear, mar, face_distance, thresholds, counters)
            frame = visualize_counting(frame, total_blinks, ear, total_mouth, mar, total_face)
            

        # Visualize results in a new window
        cv.imshow('Demo', frame)

        tm.reset()
