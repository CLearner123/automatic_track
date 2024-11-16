#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

#用户库
from visulTracking import laneDetection

if __name__ == '__main__':
    
    auto_track = laneDetection()
    
    
    image = cv2.imread("/home/mei24/catkin_ws/src/automatic_track/images/test_frame_0.jpg")
    undistorted_image = laneDetection.undistort_image(image, auto_track.mtx, auto_track.dist)
      
    # 左图梯形区域的四个端点
    src = np.float32([[152, 713], [1092, 714], [697, 420], [537, 424]])
    # 右图矩形区域的四个端点
    dst = np.float32([[300, 0], [950, 0], [950, 720], [300, 720]])
    
    warp_image= laneDetection.birdseye_transform(undistorted_image, src, dst)
   
    gray_binary = auto_track.graySelect(warp_image, thresh=160)
    hlsL_binary = auto_track.hlsLSelect(warp_image, (189, 255))

    combined_binary = np.zeros_like(hlsL_binary)
    combined_binary[(hlsL_binary == 1) | (gray_binary == 1)] = 1
    
    out_binary,left_fitx,right_fitx,ploty= auto_track.fit_polynomial(combined_binary, nwindows=10, margin=50, minpix=30)
    
    drawing_image = auto_track.drawing_in_originimage(undistorted_image, combined_binary, left_fitx, right_fitx, ploty,src,dst)
    
    drawing_image= np.hstack((drawing_image,out_binary))
    
    cv2.imshow("demo", drawing_image)
    
     # 等待按键并关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()