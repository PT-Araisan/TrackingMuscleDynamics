import cv2
import numpy as np

# 動画ファイルのパス
video_path = "assets/sample3.mp4"

# 動画の読み込み
cap = cv2.VideoCapture(video_path)

# 最初のフレームを取得
ret, frame1 = cap.read()
if not ret:
    print("動画を読み込めませんでした")
    cap.release()
    exit()

# ROIを選択
roi = cv2.selectROI("Select ROI", frame1, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")

# ROIの座標を取得
x, y, w, h = roi

# ROIのフレームをグレースケールに変換
prvs = cv2.cvtColor(frame1[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    
    next = cv2.cvtColor(frame2[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    # ファーンバック法によるオプティカルフローの計算
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 流れの大きさと角度を計算
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # HSV画像の設定
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # フレームを表示
    cv2.imshow('Optical Flow', rgb)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # 'Esc'キーで終了
        break
    elif k == ord('s'):  # 's'キーで画像保存
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)

    # 次のフレームのために更新
    prvs = next

# リソースの解放
cap.release()
cv2.destroyAllWindows()
