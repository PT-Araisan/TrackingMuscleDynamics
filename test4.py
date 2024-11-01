import cv2
import numpy as np

# 動画ファイルを読み込む
cap = cv2.VideoCapture("assets/sample3.mp4")

# 最初のフレームを読み込む
ret, frame1 = cap.read()
if not ret:
    print("動画の読み込みに失敗しました。")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# グレースケールに変換
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# HSV画像の初期化
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

# ROIの定義
#roi = (200, 50, 50, 50)  # 例: (x, y, 幅, 高さ)
roi = (286, 241, 50, 50)  # 例: (x, y, 幅, 高さ)
x, y, w, h = roi

# 動きの大きさを累積するための変数
total_motion_magnitude = 0.0

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    # グレースケールに変換
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # ROI内の動きを計算
    roi_prvs = prvs[y:y+h, x:x+w]
    roi_next = next[y:y+h, x:x+w]

    flow = cv2.calcOpticalFlowFarneback(roi_prvs, roi_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 流れの大きさと角度を計算
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 平均の大きさを計算し、累積する
    mean_mag = np.mean(mag)
    total_motion_magnitude += np.sum(mag)  # 動きの大きさを総合的に加算

    # 計算結果を出力
    print(f"平均の動きの大きさ: {mean_mag:.2f}, 動きの累積: {total_motion_magnitude:.2f}")

    # ROIのHSV画像を更新
    hsv_roi = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_roi[..., 0] = ang * 180 / np.pi / 2
    hsv_roi[..., 1] = 255
    hsv_roi[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # ROI内を元のフレームに描画
    rgb_roi = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)
    frame2_with_roi = frame2.copy()
    frame2_with_roi[y:y+h, x:x+w] = rgb_roi

    cv2.imshow('frame2', frame2_with_roi)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # 'ESC'キーで終了
        break
    elif k == ord('s'):  # 's'キーで画像を保存
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb_roi)

    prvs = next

# 動きの総合的な大きさを出力
print(f"全フレームの動きの総合的な大きさ: {total_motion_magnitude:.2f}")

cap.release()
cv2.destroyAllWindows()
