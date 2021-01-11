import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()
# 讀取圖檔
image_base_path = 'final_result/'
img_listing = list(Path(image_base_path).glob('*.png'))
for img_name in img_listing:
    img_name = str(img_name)
    img = cv2.imread(img_name)

    # 轉為灰階圖片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 計算直方圖每個 bin 的數值
    #hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = gray.ravel()
    y, x, _ = plt.hist(hist, 256, [0,256])
    #plt.show()
    var = variance_of_laplacian(gray)
    var_str = 'Blur = '+str(int(var))
    plt.text(5, y.max(), var_str, color = "r")
    
    plt.savefig(img_name[:-4]+'_hist.png')
    plt.cla()
