# 统计检测计数
def update_counts(ear, mar, face_distance, thresholds, counters):
    EAR_THRESH, EYE_close, MAR_THRESH, FACE_THRESH = thresholds
    count_eye, total_blinks, count_mouth, total_mouth, distance_left, distance_right, total_face = counters

    if ear < EAR_THRESH:
        count_eye += 1
    else:
        if count_eye >= EYE_close:
            total_blinks += 1
        count_eye = 0

    if mar > MAR_THRESH:
        count_mouth += 1
    else:
        if count_mouth >= 2:
            total_mouth += 1
        count_mouth = 0

    face_left1, face_right1, face_left2, face_right2 = face_distance
    if face_left1 >= face_right1 + FACE_THRESH and face_left2 >= face_right2 + FACE_THRESH:
        distance_left += 1
    if face_right1 >= face_left1 + FACE_THRESH and face_right2 >= face_left2 + FACE_THRESH:
        distance_right += 1
    if distance_left != 0 and distance_right != 0:
        total_face += 1
        distance_left = 0
        distance_right = 0

    return count_eye, total_blinks, count_mouth, total_mouth, distance_left, distance_right, total_face