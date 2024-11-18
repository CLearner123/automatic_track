#!/usr/bin/env python3

#该文件用于记录视频和图片

#外部库
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge , CvBridgeError
import os



    

def ImageCallback(msg):   
    try:
        cv_image = bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
    except CvBridgeError as e:
        rospy.logerr("error".format(e))
        return


        
             
        
        
if __name__ == "__main__":
    
    rospy.init_node("Image_sub")
    
    bridge =CvBridge()
      
    image_sub =rospy.Subscriber("/camera/color/image_raw",Image,ImageCallback,queue_size=10)
    
    rospy.spin()
    
    cv2.destroyAllWindows()