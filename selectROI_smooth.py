import numpy as np
import cv2

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.GaussianBlur(old_gray, (5, 5), 0)

    x, y, w, h = cv2.selectROI("ROI Selection", old_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("ROI Selection")

    roi_mask = np.zeros(old_gray.shape, dtype=np.uint8)  
    roi_mask = cv2.rectangle(roi_mask, (x, y), (x + w, y + h), (255), -1)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7, mask=roi_mask)

    x_coords, y_coords = [], []
    mask = np.zeros_like(old_frame)
    lk_params = dict(winSize=(200, 200), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
            x_coords.append(a)
            y_coords.append(b)

        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)

        key = cv2.waitKey(10)
        if key == 27:  
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()
    
    if x_coords and y_coords:
        criteria = (x_coords[0], y_coords[0])
        xx = x_coords[30] if len(x_coords) > 30 else x_coords[-1]
        yy = y_coords[30] if len(y_coords) > 30 else y_coords[-1]
        dist = np.sqrt((xx - criteria[0]) ** 2 + (yy - criteria[1]) ** 2)
        print('移動距離：{0}px'.format(dist))

video_path = 'assets/sample2.mp4'  
process_video(video_path)

