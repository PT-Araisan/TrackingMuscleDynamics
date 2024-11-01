import cv2
import numpy as np

# 動画ファイルのパス
video_path = 'assets/sample2.mp4'

# 動画の読み込み
cap = cv2.VideoCapture(video_path)

# 最初のフレームを取得
ret, old_frame = cap.read()
if not ret:
    print("動画を読み込めませんでした")
    cap.release()
    exit()

# ROIを選択
roi = cv2.selectROI("Select ROI", old_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")

# ROIの座標を取得
x, y, w, h = roi

# ROIのフレームをグレースケールに変換
old_gray = cv2.cvtColor(old_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
old_points = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# メインループ
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    # オプティカルフローの計算
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None)

    # 有効なポイントを抽出
    good_new = new_points[status == 1]
    good_old = old_points[status == 1]

    # 動きを描画
    for i in range(len(good_new)):
        a, b = good_new[i].ravel()
        c, d = good_old[i].ravel()
        cv2.line(frame, (int(a + x), int(b + y)), (int(c + x), int(d + y)), (0, 255, 0), 2)
        cv2.circle(frame, (int(a + x), int(b + y)), 5, (0, 0, 255), -1)

    # フレームを表示
    cv2.imshow('Optical Flow', frame)

    # 次のフレームのために更新
    old_gray = frame_gray.copy()
    old_points = good_new.reshape(-1, 1, 2)

    # 'q'で終了
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()

