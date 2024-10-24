#!/usr/bin/env python3

import cv2
import numpy as np
import glob

#######################相机参数############################

ret = 0   #校准成功的标志
mtx = 0   #相机内参矩阵
dist = 0  #畸变系数（描述镜头的畸变）
rvecs = 0  #旋转向量
tvecs = 0  #平移向量


######################交点坐标数量##########################

nx = 0   #x方向
ny = 0   #y方向


class CameraCalibration:
    #################################################################
    # Step 1 : 读入图片、预处理图片、检测交点、标定相机
    #################################################################
    def getCameraCalibrationCoefficients(chessboardname, nx, ny):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        
        images = glob.glob(chessboardname)
        if len(images) > 0:
            print("images num for calibration : ", len(images))
        else:
            print("No image for calibration.")
            return
        
        ret_count = 0
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = (img.shape[1], img.shape[0])
            # Finde the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                ret_count += 1
                objpoints.append(objp)
                imgpoints.append(corners)
                
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        print('Do calibration successfully')
        return ret, mtx, dist, rvecs, tvecs

    #################################################################
    # Step 2 : 传入畸变参数将畸变的图像进行畸变修正处理。
    #################################################################
    def undistortImage(distortImage, mtx, dist):
        return cv2.undistort(distortImage, mtx, dist, None, mtx)



