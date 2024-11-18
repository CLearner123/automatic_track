#!/usr/bin/env python3

#该文件用于记录视频和图片

#外部库
import rospy
from sensor_msgs.msg import Image 
from sensor_msgs.msg import Float64 
import cv2
import numpy as np
from cv_bridge import CvBridge , CvBridgeError

#用户库
from laneDetection import laneDetection


class Node:
    def __init__(self):
        
        self.LaneDetect = laneDetection()
        self.bridge =CvBridge()
        self.bias_pub = rospy.Publisher('/VisualTrack/bias', Float64, queue_size=10) 
        self.out_image_pub = rospy.Publisher('/VisualTrack/out_image', Image, queue_size= 10 )
                
        self.image_sub =rospy.Subscriber("/camera/color/image_raw",Image,self.ImageCallback,queue_size=10)    
        
      
    def ImageCallback(self,msg):   
        try:           
            cv_image = self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("error".format(e))
            return

        undistorted_image = self.LaneDetect.undistort_image(cv_image, self.LaneDetect.mtx, self.LaneDetect.dist)

        # 左图梯形区域的四个端点
        src = np.float32([[152, 713], [1092, 714], [697, 420], [537, 424]])
        # 右图矩形区域的四个端点
        dst = np.float32([[300, 0], [950, 0], [950, 720], [300, 720]])
        
        warp_image= self.LaneDetect.birdseye_transform(undistorted_image, src, dst) 
        
        gray_binary = self.LaneDetect.graySelect(warp_image, thresh=160)
        hlsL_binary = self.LaneDetect.hlsLSelect(warp_image, (189, 255))

        combined_binary = np.zeros_like(hlsL_binary)
        combined_binary[(hlsL_binary == 1) | (gray_binary == 1)] = 1
        
        out_binary,left_fitx,right_fitx,ploty= self.LaneDetect.fit_polynomial(combined_binary, nwindows=10, margin=50, minpix=30)
        
        bias = self.LaneDetect.bias_caculate(combined_binary, left_fitx, right_fitx, ploty, [0.3, 0.3, 0.3, 0.1])  
        
        # 发布偏置  
        self.bias_pub.publish(bias)
        
         # 发布二值化图像
        try:
            binary_image_msg = self.bridge.cv2_to_imgmsg((out_binary * 255).astype(np.uint8), encoding="mono8")
            self.out_image_pub.publish(binary_image_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert binary image to ROS Image: {e}")
       
             
        
        
if __name__ == "__main__":
    
    rospy.init_node("Visul_Tracking")
    rospy.loginfo("Visul Tracking started.")
    
    Visul_Tracking_node = Node()
    
    rospy.spin()
