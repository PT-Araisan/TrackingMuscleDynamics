import numpy as np
import cv2
from google.colab.patches import cv2_imshow

cap = cv2.VideoCapture('sample.mp4')

# ShiTomasiコーナー検出のためのパラメータ
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Lucas-Kanade法のパラメータ
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# カラー設定
color = np.random.randint(0, 255, (100, 3))

# ROIの設定 (100×100)
roi_x, roi_y, roi_size = 170, 300, 100
roi = (roi_x, roi_y, roi_x + roi_size, roi_y + roi_size)

# 最初のフレームを取得
ret, old_frame = cap.read()
if not ret:
    print("最初のフレームの読み込みに失敗しました。")
    cap.release()
    exit()

# ROI内のフレームを取得
old_frame_roi = old_frame[roi[1]:roi[3], roi[0]:roi[2]]
old_gray = cv2.cvtColor(old_frame_roi, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# マスクの作成
mask = np.zeros_like(old_frame)

# 動画出力の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4コーデック
output_filename = 'optical_flow_output_with_roi.mp4'
out = cv2.VideoWriter(output_filename, fourcc, 30.0, (old_frame.shape[1], old_frame.shape[0]))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI内のフレームを取得
    frame_roi = frame[roi[1]:roi[3], roi[0]:roi[2]]
    frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

    # Optical Flowの計算
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # p1がNoneでないか確認
        if p1 is None or st is None:
            print("オプティカルフロー計算に失敗しました。")
            break

        # 有効なポイントのインデックスを取得
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # ポイントを描画
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            # 座標を整数に変換
            a, b, c, d = map(int, (a + roi_x, b + roi_y, c + roi_x, d + roi_y))

            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)

        # フレームを動画に書き込む
        out.write(img)

        # 次のフレームの準備
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) if good_new.size > 0 else None
    else:
        # ROI内で再度特徴点を検出する
        old_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# リソースを解放
cap.release()
out.release()
print(f"動画ファイル {output_filename} を保存しました。")
