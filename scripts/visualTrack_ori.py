#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#File :   visual_patrol6.1.py
#Time :   2024/06/23 15:09:44
#Author:   DJZ 
#Email:   DJZ0217@163.com


import cv2
import numpy as np
import rospy
import math
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import yaml

class LaneDetection:
    def __init__(self):
        
        rospy.init_node('lane_detection', anonymous=False)
        ## 节点参数
        self.rate = rospy.Rate(30)
        rospy.loginfo("start")
        self.pub_show_line = rospy.Publisher('/lane_detection/show_lines', Image, queue_size=1)

        self.sub_image = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_msg = Twist()

        try:
            with open('/home/iflytek/task_ws/src/lane_detection/config/parameters_6_2.yaml', 'r') as file:
                parameters = yaml.safe_load(file)
            self.gamma = parameters['gamma']
            self.thresh = parameters['thresh']
            self.kp = parameters['kp']
            self.ki = parameters['ki']
            self.kd = parameters['kd']
            self.speed = parameters['speed']
            self.signal_buffer = parameters['signal_buffer']
            self.p1 = parameters['p1']
            self.p2 = parameters['p2']
            self.p3 = parameters['p3']
            self.p4 = parameters['p4']
            self.speed_0 = parameters['speed_0']
            self.kp_0 = parameters['kp_0']
            self.ki_0 = parameters['ki_0']
            self.kd_0 = parameters['kd_0']
            self.p1_0 = parameters['p1_0']
            self.p2_0 = parameters['p2_0']
            self.p3_0 = parameters['p3_0']
            self.p4_0 = parameters['p4_0']            
            self.target = 0
            self.error = 0
            self.current_pos = 320
            self.prev_error = 0
            self.integral = 0
            self.lane_flag = 0
            self.turn_speed = 0.30

        except:
            rospy.loginfo("load parameters error")
            exit(0)
    
    """
    去除图像畸变

    参数:
        image: 输入图像
        camera_matrix: 相机内参数矩阵
        distortion_coeffs: 畸变系数

    返回:
        去除畸变后的图像
    """
    def undistort_image(self,image, camera_matrix, distortion_coeffs):
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
    def birdseye_transform(self,image, src_points, dst_points):

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # 应用透视变换
        birdseye_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return birdseye_image
    
    '''
    提取白色区域

    参数:
        image: 输入图像
        thresh: 阈值
    返回:
        二值化图像
    '''
    def GraySelect(self,img, thresh=150):
        
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_white = cv2.inRange(gray_image, thresh, 255)
        # ret, binary = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)
        binary_image = mask_white
        return binary_image
    

    
    '''
    gamma校正,用于滤除反光
    '''
    def gamma_correction(self,image, gamma=2.0):
        # 将图像转换为浮点数类型
        image = np.float32(image) / 255.0
        # 对图像进行伽马校正
        corrected_image = np.power(image, gamma)
        # 将图像转换回uint8类型（0-255）
        corrected_image = np.uint8(corrected_image * 255.0)
        return corrected_image
    
    '''
    进行多项式拟合
    '''
    def curve_fitting(self, x_coords, y_coords, degree=2):
        # 将x_coords和y_coords转换为numpy数组
        x = np.array(x_coords)
        y = np.array(y_coords)

        # 进行多项式拟合，degree是拟合的多项式的最高次幂
        coefficients = np.polyfit(x, y, degree)
        return coefficients
    
    
    '''
    寻找车道线,并计算中心轨迹线
    '''
    def find_line(self, image,draw_image):
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_contours = [x for x in contours if len(x) > 80 and cv2.contourArea(x) > 1000]
        # print(f"Number of contours: {len(new_contours)}")
        left_points = []
        right_points = []
        for contour in new_contours:
            # print(f"length of contour: {len(contour)}")
            # 绘制轮廓
            cv2.drawContours(draw_image, [contour], -1, (0, 200, 0), 2)
            contour_sorted = sorted(contour, key=lambda point: point[0][1], reverse=True)
            if self.lane_flag == 0:
                if contour_sorted[0][0][0] < 320:
                    left_points.append(contour_sorted)
                else:
                    right_points.append(contour_sorted)
            elif self.lane_flag == 1:
                if len(new_contours) == 1:
                    left_points.append(contour_sorted)
                else:
                    if contour_sorted[0][0][0] < 320:
                        left_points.append(contour_sorted)
                    else:
                        right_points.append(contour_sorted)
            elif self.lane_flag == 2:
                if len(new_contours) == 1:
                    right_points.append(contour_sorted)
                else:
                    if contour_sorted[0][0][0] < 320:
                        left_points.append(contour_sorted)
                    else:
                        right_points.append(contour_sorted)
        if left_points and right_points:
            self.lane_flag = 0
        elif left_points and not right_points:
            self.lane_flag = 1
        elif not left_points and right_points:
            self.lane_flag = 2         
                
        # 分别存储left_points的x和y坐标
        left_x_coords = [0]
        left_y_coords = [0]
        for contour in left_points:
            for point in contour:
                left_x_coords.append(point[0][0])
                left_y_coords.append(point[0][1])   
        # 分别存储right_points的x和y坐标
        right_x_coords = [0]
        right_y_coords = [0]
        for contour in right_points:
            for point in contour:
                right_x_coords.append(point[0][0])
                right_y_coords.append(point[0][1])
        if len(left_y_coords)>1:
            left_line = self.curve_fitting(left_y_coords, left_x_coords,2)
        if len(right_y_coords)>1:
            right_line = self.curve_fitting(right_y_coords, right_x_coords,2)
        left_max = max(left_y_coords)
        left_low = min(left_y_coords)
        if left_low==0 and len(left_y_coords)>1:
            left_low = min(left_y_coords[1:])
        right_max = max(right_y_coords)
        right_low = min(right_y_coords)
        if right_low == 0 and len(right_y_coords) >1:
            right_low = min(right_y_coords[1:])
        center_line_x = []
        center_line_y = []

        for i in range(0,170,2):
            flag_left = (i-left_max)*(i-left_low)
            flag_right = (i-right_max)*(i-right_low)
            if flag_left < 0 and flag_right < 0:
                x_left = int(left_line[0]*i**2 + left_line[1]*i + left_line[2])
                x_right = int(right_line[0]*i**2 + right_line[1]*i + right_line[2])
            elif flag_left < 0 and flag_right > 0 :
                x_left = int(left_line[0]*i**2 + left_line[1]*i + left_line[2])
                if left_max > 500 and left_low < 40:
                    x_right = 800
                else:
                    x_right = 640
            elif flag_left > 0 and flag_right < 0:
                x_right = int(right_line[0]*i**2 + right_line[1]*i + right_line[2])
                if right_max > 500 and right_low < 40:
                    x_left = -320
                else:
                    x_left = 0
            else:
                continue
            x = int((x_left+x_right)/2)  
            center_line_x.append(x)
            center_line_y.append(i)
            cv2.circle(draw_image, (x_left, i), 1, (255, 0,0), -1)
            cv2.circle(draw_image, (x_right, i), 1, (255, 0,0), -1)
            cv2.circle(draw_image, (x, i), 1, (0, 0, 255), -1)
        return center_line_x,center_line_y
   
    # PID控制器函数
    def pid_controller(self,dt,kp,ki,kd):
        self.error = (self.target)/100
        self.integral += self.error * dt
        derivative = (self.error - self.prev_error) / dt
        output = -kp * self.error + ki * self.integral + kd * derivative
        self.prev_error = self.error
        return output
    
    def car_controller(self,dt,speed,kp,ki,kd):
        control_signal = self.pid_controller(dt,kp,ki,kd)
        # 限制控制信号的范围，防止过大或过小的输出
        control_signal = max(min(control_signal,0.5), -0.5)
        cmd_vel = Twist()
        if control_signal == 0 :
            control_signal = 0.0001
        R = 0.115/math.tan(control_signal)
        W = speed/R
        cmd_vel.linear.x = speed
        cmd_vel.angular.z = W  # Assuming only controlling linear velocity
        self.cmd_vel_pub.publish(cmd_vel)
        rospy.loginfo('control_signal: %f', control_signal)
        return True
    
    def image_callback(self, msg):
        prev_time = rospy.get_time()
        frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        input_image = frame
        gamma_image = self.gamma_correction(input_image, self.gamma)
        target_img = gamma_image[250:420, 0:640]
        gray_image = self.GraySelect(target_img, self.thresh)
        center_x,_ = self.find_line(gray_image,target_img)
        current_time = rospy.get_time()
        dt = current_time - prev_time
        length = len(center_x)
        self.target = self.p1 * (center_x[0] - 320) + self.p2* (sum(center_x[int(length/2):]) / (2*length) - 320) + self.p3* (center_x[2] - center_x[-2]) + self.p4*(sum(center_x[:int(length/2)])/(2*length) - 320)
        if abs(self.target) > self.signal_buffer:
            self.target = self.p1_0 * (center_x[0] - 320) + self.p2_0* (sum(center_x[int(length/2):]) / (2*length) - 320) + self.p3_0* (center_x[2] - center_x[-2]) + self.p4_0*(sum(center_x[:int(length/2)])/(2*length) - 320)
            self.car_controller(dt,self.speed_0,self.kp_0,self.ki_0,self.kd_0)
        else:
            self.car_controller(dt,self.speed,self.kp,self.ki,self.kd)
        self.pub_show_line.publish(CvBridge().cv2_to_imgmsg(target_img, "bgr8"))
        elapsed_time = dt
        rospy.loginfo(f"\n\nElapsed time: {elapsed_time:.3f} seconds, {(1.0/elapsed_time):.3f}fps\n\n")
        
        
    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == "__main__":
    try:
        lane_detection = LaneDetection()
        lane_detection.run()
    except rospy.ROSInterruptException:
        exit(0)
