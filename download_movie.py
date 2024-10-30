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

# 最初のフレームを取得
ret, old_frame = cap.read()
if not ret:
    print("最初のフレームの読み込みに失敗しました。")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# マスクの作成
mask = np.zeros_like(old_frame)

# 動画出力の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4コーデック
output_filename = 'optical_flow_output.mp4'
out = cv2.VideoWriter(output_filename, fourcc, 30.0, (old_frame.shape[1], old_frame.shape[0]))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Optical Flowの計算
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # 有効なポイントのインデックスを取得
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # ポイントを描画
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        
        # 座標を整数に変換
        a, b, c, d = map(int, (a, b, c, d))
        
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    
    img = cv2.add(frame, mask)

    # フレームを動画に書き込む
    out.write(img)

    # cv2_imshow(img)  # 表示する場合はこれをコメントアウト

    # 次のフレームの準備
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# リソースを解放
cap.release()
out.release()
print(f"動画ファイル {output_filename} を保存しました。")
