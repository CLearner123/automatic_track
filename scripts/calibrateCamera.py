#!/usr/bin/env python3

import cv2
import numpy as np
import glob
import yaml
import os




def yaml_file_path(filename):  # 获取yaml配置文件路径
    
    #获取当前脚本所在的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上一级目录  
    parent_dir = os.path.dirname(current_dir)

    yaml_path = os.path.join(parent_dir, "config", filename)
    # with open(yaml_path) as file:
    #     config = yaml.safe_load(file)
    return yaml_path
    
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
        if ret is not None :
            print('Do calibration successfully')
            return ret, mtx, dist, rvecs, tvecs
        else:
            print('Calibration failed.')
            return None, None, None, None, None
    

    #################################################################
    # Step 2 : 传入畸变参数将畸变的图像进行畸变修正处理。
    #################################################################
    def undistortImage(distortImage, mtx, dist):
        return cv2.undistort(distortImage, mtx, dist, None, mtx)


if __name__ == '__main__':
    
    yaml_file_path = yaml_file_path("Intel_D455.yaml")
    
    with open(yaml_file_path) as file:
        camera_config = yaml.safe_load(file)
        
    nx = camera_config["chessboard_pattern"]["nx"]
    
    ny = camera_config["chessboard_pattern"]["ny"]
   
    ret, mtx, dist, rvecs, tvecs = CameraCalibration.getCameraCalibrationCoefficients('src/automatic_track/images/calibration*.jpg', nx, ny)

    camera_config["calibration_pattern"]["ret"] = ret
    camera_config["calibration_pattern"]["mtx"] = np.vstack(mtx).tolist()
    camera_config["calibration_pattern"]["dist"] = np.vstack(dist).tolist()
    camera_config["calibration_pattern"]["rvecs"] = [rvec.flatten().tolist() for rvec in rvecs]
    camera_config["calibration_pattern"]["tvecs"] = [tvec.flatten().tolist() for tvec in tvecs]
    
    with open(yaml_file_path, 'w') as outfile:
        yaml.dump(camera_config, outfile, default_flow_style=False)
        
    
    # 继续处理 ret, mtx, dist, rvecs, tvecs
    print("Calibration results:")
    print("ret:", ret)
    print("mtx:", mtx)
    print("dist:", dist)
    print("rvecs:", rvecs)
    print("tvecs:", tvecs)
    
    
    # image = cv2.imread("/home/mei24/catkin_ws/src/automatic_track/images/calibration_1730725873.6874537.jpg")
    # undistorted_image = CameraCalibration.undistortImage(image, mtx, dist)
    # image = np.hstack((image,undistorted_image))
    # cv2.imshow("Undistorted Image", image)
    
    #  # 等待按键并关闭窗口
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

