import cv2
import numpy as np

def ColorCorrect1(img,u):
    img = np.float64(img) / 255
    B_mse = np.std(img[:,:,0])
    G_mse = np.std(img[:,:,1])
    R_mse = np.std(img[:,:,2])

    B_max = np.mean(img[:,:,0]) + u * B_mse
    G_max = np.mean(img[:,:,1]) + u * G_mse
    R_max = np.mean(img[:,:,2]) + u * R_mse

    B_min = np.mean(img[:,:,0]) - u * B_mse
    G_min = np.mean(img[:,:,1]) - u * G_mse
    R_min = np.mean(img[:,:,2]) - u * R_mse

    B_cr = (img[:,:,0]  - B_min) / (B_max - B_min)
    G_cr = (img[:,:,1]  - G_min) / (G_max - G_min)
    R_cr = (img[:,:,2]  - R_min) / (R_max - R_min)
    
    img_CR = cv2.merge([B_cr,G_cr,R_cr]) *255
    img_CR = np.clip(img_CR,0,255)
    img_CR = np.uint8(img_CR)

 

    return img_CR
     
