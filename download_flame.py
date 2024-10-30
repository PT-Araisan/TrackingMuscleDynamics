import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import os

# 動画ファイルの読み込み
cap = cv2.VideoCapture('sample.mp4')

# 保存先のフォルダを作成
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# フレーム数を取得
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_number in range(total_frames):
    # 指定したフレームを取得
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if ret:      
        # フレームをPNG画像として保存
        output_path = os.path.join(output_folder, f'frame_{frame_number + 1}.png')
        cv2.imwrite(output_path, frame)
        print(f"フレームを保存しました: {output_path}")
    else:
        print(f"フレーム {frame_number + 1} の読み込みに失敗しました。")

# リソースを解放
cap.release()

import shutil
from google.colab import files

# 出力フォルダのパス
output_folder = 'output_images'
# フォルダをZIPファイルに圧縮
shutil.make_archive(output_folder, 'zip', output_folder)
# ZIPファイルをダウンロード
files.download(f'{output_folder}.zip')
