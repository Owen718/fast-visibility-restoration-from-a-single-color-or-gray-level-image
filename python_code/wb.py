import cv2
import numpy as np
def white_balance_1(img):
    '''
    第一种简单的求均值白平衡法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    # 读取图像
    r, g, b = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])
    return balance_img


 
def white_balance_2(img_input):
    '''
    完美反射白平衡
    STEP 1：计算每个像素的R\G\B之和
    STEP 2：按R+G+B值的大小计算出其前Ratio%的值作为参考点的的阈值T
    STEP 3：对图像中的每个点，计算其中R+G+B值大于T的所有点的R\G\B分量的累积和的平均值
    STEP 4：对每个点将像素量化到[0,255]之间
    依赖ratio值选取而且对亮度最大区域不是白色的图像效果不佳。
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    img = img_input.copy()
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    for i in range(m):
        for j in range(n):
            sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1
 
    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    for i in range(m):
        for j in range(n):
            if sum_[i][j] >= key:
                sum_b += b[i][j]
                sum_g += g[i][j]
                sum_r += r[i][j]
                time = time + 1
 
    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time
 
    maxvalue = float(np.max(img))
    # maxvalue = 255
    for i in range(m):
        for j in range(n):
            b = int(img[i][j][0]) * maxvalue / int(avg_b)
            g = int(img[i][j][1]) * maxvalue / int(avg_g)
            r = int(img[i][j][2]) * maxvalue / int(avg_r)
            if b > 255:
                b = 255
            if b < 0:
                b = 0
            if g > 255:
                g = 255
            if g < 0:
                g = 0
            if r > 255:
                r = 255
            if r < 0:
                r = 0
            img[i][j][0] = b
            img[i][j][1] = g
            img[i][j][2] = r
 
    return img
 
def white_balance_3(img):
    '''
    灰度世界假设
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    B, G, R = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)
 
    for i in range(len(Ba)):
        for j in range(len(Ba[0])):
            Ba[i][j] = 255 if Ba[i][j] > 255 else Ba[i][j]
            Ga[i][j] = 255 if Ga[i][j] > 255 else Ga[i][j]
            Ra[i][j] = 255 if Ra[i][j] > 255 else Ra[i][j]
 
    # print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
    dst_img = np.uint8(np.zeros_like(img))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra
    return dst_img
