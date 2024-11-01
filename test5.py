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
roi = (286, 241, 50, 50)  # (x, y, 幅, 高さ)
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
    print(f" 動きの累積: {total_motion_magnitude:.2f}")

    # 矢印を描画
    for y_pos in range(0, h, 5):  # 矢印を描画する間隔
        for x_pos in range(0, w, 5):
            if mag[y_pos, x_pos] > 1:  # 動きが一定以上のときだけ描画
                cv2.arrowedLine(frame2, 
                                (x + x_pos, y + y_pos), 
                                (x + x_pos + int(flow[y_pos, x_pos, 0]), 
                                 y + y_pos + int(flow[y_pos, x_pos, 1])), 
                                (0, 255, 0), 1, tipLength=1)  # 太さと先端のサイズを調整

    # ROIを元のフレームに描画
    frame2_with_roi = frame2.copy()
    cv2.rectangle(frame2_with_roi, (x, y), (x+w, y+h), (255, 0, 0), 2)  # ROIを青で描画

    cv2.imshow('frame2', frame2_with_roi)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # 'ESC'キーで終了
        break
    elif k == ord('s'):  # 's'キーで画像を保存
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', frame2_with_roi)

    prvs = next

# 動きの総合的な大きさを出力
print(f"全フレームの動きの総合的な大きさ: {total_motion_magnitude:.2f}")

cap.release()
cv2.destroyAllWindows()


