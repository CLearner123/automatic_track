#!/usr/bin/env python3

import cv2
import numpy as np
import glob
import yaml
import os
import matplotlib.pyplot as plt

#用户库
import calibrateCamera




class laneDetection():

    def __init__(self):
        
        
        
        yaml_file_path = calibrateCamera.yaml_file_path("ost.yaml")
    
        with open(yaml_file_path) as file:
            camera_config = yaml.safe_load(file)  
            
        # ret = camera_config["calibration_pattern"]["ret"]
        self.mtx = np.array(camera_config["camera_matrix"]["data"], dtype=np.float32).reshape((3, 3))
        self.dist = np.array(camera_config["distortion_coefficients"]["data"])
        # rvecs = [np.array(rvec).reshape(-1, 1) for rvec in  camera_config["calibration_pattern"]["rvecs"]]
        # tvecs = [np.array(tvec).reshape(-1, 1) for tvec in  camera_config["calibration_pattern"]["tvecs"]]
        
        print("Calibration results:")
        # print("ret:", ret)
        print("mtx:", self.mtx)
        print("dist:", self.dist)
        # print("rvecs:", rvecs)
        # print("tvecs:", tvecs)
        
        pass
    
    
    
    
    """
    去除图像畸变

    参数:
        image: 输入图像
        camera_matrix: 相机内参数矩阵
        distortion_coeffs: 畸变系数

    返回:
        去除畸变后的图像
    """
    def undistort_image(image, camera_matrix, distortion_coeffs):
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 0)
        undistorted_image = cv2.undistort(image, camera_matrix, distortion_coeffs, None, new_camera_matrix)
        return undistorted_image
    
    """
    进行俯视投影变换(透视变换)

    参数:
        image: 输入图像
        src_points: 原始图像中的四个顶点坐标
        dst_points: 目标俯视图像中的四个顶点坐标

    返回:
        俯视投影变换后的图像
    """
    def birdseye_transform(image, src_points, dst_points):

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points) #正变换矩阵
        Minv = cv2.getPerspectiveTransform(dst_points, src_points)  #
        # 应用透视变换
        birdseye_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        
        return birdseye_image
    
    """
     进行逆俯视投影变换(透视变换)

    参数:
        birdseye_image: 鸟瞰图
        src_points: 原始图像中的四个顶点坐标
        dst_points: 目标俯视图像中的四个顶点坐标

    返回:
        俯视投影变换前的图像 
                 
    """   
    def inverse_birdseye_transform(birdseye_image, src_points, dst_points):

        # 计算逆透视变换矩阵
        Minv = cv2.getPerspectiveTransform(dst_points, src_points)  # 从目标点到源点
        
        # 应用逆透视变换
        originalview_image = cv2.warpPerspective(birdseye_image, Minv, (birdseye_image.shape[1], birdseye_image.shape[0]), flags=cv2.INTER_LINEAR)
        
        return originalview_image

    
    """
    回调函数，用于捕捉鼠标点击时的坐标  
     
    """
    def get_pixel_position(self,event, x, y,flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            print(f"Pixel position: ({x}, {y})")
            
            
    """
    用于获得图片中的src_point
    
    参数:image
       
    """        
    def get_src_point(self,image):
        
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.get_pixel_position, image)
        cv2.imshow("image",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    '''
    soble算子检测边线,但是好像没有用
    

    
    '''          
    def absSobelThreshold(self,img, orient, thresh_min, thresh_max):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Return the result
        return binary_output
        
    '''
    提取白色区域

    参数:
        image: 输入图像
        thresh: 阈值
    返回:
        二值化图像
    '''
    def graySelect(self,img, thresh=150):
        
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_white = cv2.inRange(gray_image, thresh, 255)
        # ret, binary = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)
        binary_image = mask_white
        return binary_image
    '''
    通过转化HSL空间提取白色车道
    
    参数：
        image: 输入图像
        thresh: 白色阈值
    
    '''
    def hlsLSelect(self,img, thresh=(220, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:,:,1]
        l_channel = l_channel*(255/np.max(l_channel))
        binary_output = np.zeros_like(l_channel)
        binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
        return binary_output
    
    '''
    在二进制透视图中识别和提取左右车道的像素位置
    
    参数:
        binary_warped: 二进制透视图
        nwindows: 窗口数量
        margin: 窗口边界
        minpix: 最小像素数量
    
    返回:
        左右车道的像素位置
      
    '''
    def find_lane_pixels(self,binary_warped, nwindows, margin, minpix):
        # 绘制下半部分直方图
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # 叠加成一个RGB图
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # 找出直方图左右两半峰值
        # 这将是左右边线的起点
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # 设置窗口的高度取决于nwindows和图像形状
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # 确定图像中所有非零像素的 x 和 y 位置
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # 当前的位置会随着每个窗口更新
        leftx_current = leftx_base
        rightx_current = rightx_base

        # 创建空列表以接收左右车道像素索引
        left_lane_inds = []
        right_lane_inds = []

        # 一个一个窗口步进
        for window in range(nwindows):
            # 分别确定左右x和y窗口边界
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # # 在图像上画出可视窗格
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            # (win_xleft_high,win_y_high),(0,255,0), 2) 
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),
            # (win_xright_high,win_y_high),(0,255,0), 2) 
            
            #确定窗格内x方向和y方向的非零像素#
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # 添加这些非零像素到列表
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            #如果发现 > minpix ，则根据它们的平均位置重新调整下一个窗口的位置
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img
    '''
    画出两侧车道所有像素和车道线
    
    参数:
        binary_warped: 二进制透视图
        nwindows: 窗口数量
        margin: 窗口边界
        minpix: 最小像素数量
        
    返回：
        out_img:画完的图像
        left_fitx:左车道的x坐标
        right_fitx:右车道的x坐标
        ploty:离散的y坐标
    
    
    '''
    def fit_polynomial(self,binary_warped, nwindows=9, margin=60, minpix=50):
        #获取左右车道像素点
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(
            binary_warped, nwindows, margin, minpix)

        # 拟合二次曲线
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            #都是x关于y的函数
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
      
        #使用cv2画出车道线
        out_img[ploty.astype(int), left_fitx.astype(int)] = [0, 255, 0]
        out_img[ploty.astype(int), right_fitx.astype(int)] = [0, 255, 0]
        # 创建一个空图像
        out_img = out_img.astype(np.uint8)
        window_img = np.zeros_like(out_img, dtype=np.uint8)
          
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 30))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 30))

        out_img = cv2.addWeighted(out_img, 1, window_img,0.1,0)
        

        return out_img, left_fitx, right_fitx, ploty
    
    '''
    将得到的车道画在原图上
    
    '''
    def drawing_in_originimage(self,undist_img, bin_warped, left_fitx, right_fitx,ploty,src_points,dst_points):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(bin_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # 计算逆透视变换矩阵
        Minv = cv2.getPerspectiveTransform(dst_points, src_points)  # 从目标点到源点
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist_img.shape[1], undist_img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
        return result
    
    
    
    
    
if __name__ == '__main__':
    
    auto_track = laneDetection()
    
    
    image = cv2.imread("/home/mei24/catkin_ws/src/automatic_track/images/saved_frame_5.jpg")
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
    
    cv2.imshow("out_binary", drawing_image)
    
     # 等待按键并关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
   

