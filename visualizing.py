import cv2 as cv
import numpy as np
from extracting import extract_palm_roi

def visualize_hand_detection(image, results, print_results=False, fps=None):
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
        for p in palm_landmarks[:5]:
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

# 显示检测结果
def visualize_counting(frame, total_blinks, ear, total_mouth, mar, total_face):
    cv.putText(frame, "Blinks: {}".format(total_blinks), (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(frame, "Mouth is open: {}".format(total_mouth), (10, 60),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(frame, "MAR: {:.2f}".format(mar), (300, 60),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(frame, "Shake one's head: {}".format(total_face), (10, 90),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame
