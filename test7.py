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

# ROIの定義
roi1 = (196, 100, 50, 50)  # 第一のROI (x, y, 幅, 高さ)
roi2 = (196, 150, 50, 50)  # 第二のROI (x, y, 幅, 高さ) - 直下に配置
x1, y1, w1, h1 = roi1
x2, y2, w2, h2 = roi2

# 動きの大きさを累積するための変数
total_motion_magnitude1 = 0.0
total_motion_magnitude2 = 0.0

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    # グレースケールに変換
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # ROI内の動きを計算
    roi_prvs1 = prvs[y1:y1+h1, x1:x1+w1]
    roi_next1 = next[y1:y1+h1, x1:x1+w1]
    
    roi_prvs2 = prvs[y2:y2+h2, x2:x2+w2]
    roi_next2 = next[y2:y2+h2, x2:x2+w2]

    flow1 = cv2.calcOpticalFlowFarneback(roi_prvs1, roi_next1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow2 = cv2.calcOpticalFlowFarneback(roi_prvs2, roi_next2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 流れの大きさと角度を計算
    mag1, ang1 = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
    mag2, ang2 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])

    # 平均の大きさを計算し、累積する
    total_motion_magnitude1 += np.sum(mag1)  # 第一のROIの動きの大きさを加算
    total_motion_magnitude2 += np.sum(mag2)  # 第二のROIの動きの大きさを加算

    # 矢印を描画
    for y_pos in range(0, h1, 5):  # 第一のROI
        for x_pos in range(0, w1, 5):
            if mag1[y_pos, x_pos] > 1:  # 動きが一定以上のときだけ描画
                cv2.arrowedLine(frame2, 
                                (x1 + x_pos, y1 + y_pos), 
                                (x1 + x_pos + int(flow1[y_pos, x_pos, 0]), 
                                 y1 + y_pos + int(flow1[y_pos, x_pos, 1])), 
                                (0, 255, 0), 3, tipLength=0.5)  # 第一のROIの矢印

    for y_pos in range(0, h2, 5):  # 第二のROI
        for x_pos in range(0, w2, 5):
            if mag2[y_pos, x_pos] > 1:  # 動きが一定以上のときだけ描画
                cv2.arrowedLine(frame2, 
                                (x2 + x_pos, y2 + y_pos), 
                                (x2 + x_pos + int(flow2[y_pos, x_pos, 0]), 
                                 y2 + y_pos + int(flow2[y_pos, x_pos, 1])), 
                                (255, 0, 0), 3, tipLength=0.5)  # 第二のROIの矢印

    # ROIを元のフレームに描画
    frame2_with_roi = frame2.copy()
    cv2.rectangle(frame2_with_roi, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)  # 第一のROIを青で描画
    cv2.rectangle(frame2_with_roi, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)  # 第二のROIを緑で描画

    cv2.imshow('frame2', frame2_with_roi)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # 'ESC'キーで終了
        break
    elif k == ord('s'):  # 's'キーで画像を保存
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', frame2_with_roi)

    prvs = next

# 動きの総合的な大きさを出力
print(f"上のROIの動きの大きさ: {total_motion_magnitude1:.2f}")
print(f"下ROIの動きの大きさ: {total_motion_magnitude2:.2f}")

# トータルの動きの比を計算
if total_motion_magnitude2 > 0:
    total_motion_ratio = total_motion_magnitude1 / total_motion_magnitude2
else:
    total_motion_ratio = float('inf')  # ゼロ除算の処理

print(f"上の動きと下の動きの比: {total_motion_ratio:.2f}")

cap.release()
cv2.destroyAllWindows()

