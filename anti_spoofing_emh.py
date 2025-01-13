from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np

# 计算眼睛的纵横比（EAR）
def EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# 计算嘴巴的纵横比（MAR）
def MAR(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])
    B = np.linalg.norm(mouth[4] - mouth[7])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

# 计算鼻子到左右脸边界的欧式距离
def nose_jaw_distance(nose, jaw):
    face_left1 = dist.euclidean(nose[0], jaw[0])
    face_right1 = dist.euclidean(nose[0], jaw[16])
    face_left2 = dist.euclidean(nose[3], jaw[2])
    face_right2 = dist.euclidean(nose[3], jaw[14])
    return (face_left1, face_right1, face_left2, face_right2)

def main():
    # 初始化参数
    EAR_THRESH = 0.3
    EYE_close = 2
    MAR_THRESH = 0.5
    FACE_THRESH = 2

    # 初始化计数器
    count_eye = 0
    total_blinks = 0
    count_mouth = 0
    total_mouth = 0
    distance_left = 0
    distance_right = 0
    total_face = 0

    # 加载人脸检测器和关键点预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 定义眼睛、嘴巴和下巴的关键点索引
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    # 启动视频流
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 在灰度图中检测人脸
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # 提取眼睛、嘴巴和下巴的坐标
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            jaw = shape[jStart:jEnd]

            # 计算眼睛的纵横比
            ear_left = EAR(leftEye)
            ear_right = EAR(rightEye)
            ear = (ear_left + ear_right) / 2.0

            # 计算嘴巴的纵横比
            mar = MAR(mouth)

            # 计算鼻子到左右脸边界的欧式距离
            face_distance = nose_jaw_distance(shape[27:31], shape[0:17])
            face_left1, face_right1, face_left2, face_right2 = face_distance

            # 框选人脸
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 眨眼检测
            if ear < EAR_THRESH:
                count_eye += 1
            else:
                if count_eye >= EYE_close:
                    total_blinks += 1
                count_eye = 0

            # 张嘴检测
            if mar > MAR_THRESH:
                count_mouth += 1
            else:
                if count_mouth >= 2:
                    total_mouth += 1
                count_mouth = 0

            # 摇头检测
            if face_left1 >= face_right1 + FACE_THRESH and face_left2 >= face_right2 + FACE_THRESH:
                distance_left += 1
            if face_right1 >= face_left1 + FACE_THRESH and face_right2 >= face_left2 + FACE_THRESH:
                distance_right += 1
            if distance_left != 0 and distance_right != 0:
                total_face += 1
                distance_left = 0
                distance_right = 0

            # 显示结果
            cv2.putText(frame, "Blinks: {}".format(total_blinks), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Mouth is open: {}".format(total_mouth), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Shake one's head: {}".format(total_face), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示图像
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    main()