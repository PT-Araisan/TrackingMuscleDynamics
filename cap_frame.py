import numpy as np
import cv2
import os

# 動画ファイルの読み込み
cap = cv2.VideoCapture('assets/sample.mp4')

# 保存先のフォルダを作成
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# 1フレーム目を取得
ret, frame = cap.read()

if ret:      
    # フレームをPNG画像として保存
    output_path = os.path.join('frame_1.png')
    cv2.imwrite(output_path, frame)
    print(f"フレームを保存しました: {output_path}")
else:
    print("フレーム 1 の読み込みに失敗しました。")

# リソースを解放
cap.release()
