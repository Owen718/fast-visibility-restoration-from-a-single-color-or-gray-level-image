import TM
import numpy as np
import os
import cv2
import wb
from ColorCorrect import ColorCorrect1
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


file_path = r'D:\haze dataset\O-HAZE\# O-HAZY NTIRE 2018\hazy'
#file_path = r'D:\github\fast-visibility-restoration-from-a-single-color-or-gray-level-image\test_image'
file_list = os.listdir(file_path)

s_v = 41
p=0.95

for file_name in file_list:
    original = cv2.imread(os.path.join(file_path, file_name))
    original = cv2.resize(original,dsize=None,dst=None,fx=0.2,fy=0.2)
    original_wb = wb.white_balance_2(original)

    W = TM.getMinChannel_new(original_wb)
    #cv_show('W',np.uint8(W))

    A = cv2.medianBlur(np.uint8(W),s_v)
    B = W - A
    B = np.abs(B)
    B = A - cv2.medianBlur(np.uint8(B),s_v)
    max_255_img = np.ones(B.shape,dtype = np.uint8 ) * 255 
    min_t = cv2.merge([np.uint8(p*B),np.uint8(W),max_255_img])
    min_t = TM.getMinChannel_new(min_t)
    min_t[min_t<0] = 0
    V = np.uint8(min_t)
    V = cv2.blur(V,(5,5)) #平滑滤波
    
    cv2.imwrite('result_image//V_'+file_name,V)
    #cv_show('V',np.uint8(V))

    V = np.float32(V) / 255

    R_dehazy = np.zeros((V.shape[0],V.shape[1],3), dtype=np.float32)

    original_wb = np.float32(original_wb) / 255

    for i in range(0,3,1):
        R_dehazy[:,:,i] = (original_wb[:,:,i] - V) / (1 - V)
    R_dehazy = R_dehazy / R_dehazy.max()

    R_dehazy = np.clip(R_dehazy,0,1)
    R_dehazy = np.uint8(R_dehazy*255)
    #R_dehazy = ColorCorrect1(R_dehazy,2.2)

    src = original
    h, w = src.shape[:2]
    result = np.zeros([h, w*2, 3], dtype=src.dtype)
    result[0:h,0:w,:] = original
    result[0:h,w:2*w,:] = R_dehazy
    cv2.putText(result, "original", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 2)
    cv2.putText(result, "dehazy", (w+10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 2)
    #cv_show('dehazy',result)

    cv2.imwrite('result_image//'+file_name,result)