import numpy as np
import cv2

def initialize_video_capture(video_path):
    return cv2.VideoCapture(video_path)

def track_optical_flow(cap):
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    roi = cv2.selectROI(windowName="roi", img=old_frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi

    empty_img = np.zeros_like(old_gray)
    roi_mask = cv2.rectangle(img=empty_img, pt1=(x, y), pt2=(x + w, y + h), color=(255), thickness=-1)

    feature_params = dict(maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7, mask=roi_mask)
    p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)

    mask = np.zeros_like(old_frame)
    x, y = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            x.append(a)
            y.append(b)

        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(30) & 0xFF == 27:  # ESCキーで終了
            break

    return (x[0], y[0]), (x[30], y[30]) if len(x) > 30 else (0, 0)

def calc_dist(criteria, new_pos):
    dist = np.sqrt((new_pos[0] - criteria[0]) ** 2 + (new_pos[1] - criteria[1]) ** 2)
    print(f'移動距離：{dist}px')

cap = initialize_video_capture('assets/sample2.mp4')
criteria, new_pos = track_optical_flow(cap)
calc_dist(criteria, new_pos)

cv2.destroyAllWindows()
cap.release()
