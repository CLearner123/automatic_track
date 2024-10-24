#!/usr/bin/env python3

#外部库
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge , CvBridgeError
import os



video_save = False
video_out = None
save_path = ""


def ImageCallback(msg):
    global video_out, video_save,save_path
   
    try:
        cv_image = bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
    except CvBridgeError as e:
        rospy.logerr("error".format(e))
        return

    cv2.imshow("camera_image",cv_image)
    
    if video_save and video_out is not None:
        video_out.write(cv_image)
       
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        
        rospy.signal_shutdown('user request shutdown.') 
        
    elif key == ord('s'):
        
        # 获取当前脚本所在的目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取上一级目录路径并创建 pictures 文件夹路径
        
        parent_dir = os.path.dirname(current_dir)
        
        pictures_dir = os.path.join(parent_dir, "images")
        
        # 如果 pictures 文件夹不存在，则创建
        if not os.path.exists(pictures_dir):
            os.makedirs(pictures_dir)
        
        save_path = os.path.join(pictures_dir, "calibration_" + str(rospy.get_time()) + ".jpg")
        
               
        cv2.imwrite(save_path, cv_image)
        
        rospy.loginfo("Image saved to {}".format(save_path))
    elif key ==  ord('v'):
               
        if not video_save:
            #获取当前脚本所在的目录路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 获取上一级目录路径并创建 video 文件夹路径   
            parent_dir = os.path.dirname(current_dir)
            
            video_dir = os.path.join(parent_dir, "video")
            
            # 如果 video 文件夹不存在，则创建
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            
            save_path = os.path.join(video_dir, "video_" + str(rospy.get_time()) + ".mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'H264')   #必须用这个编码格式
            
            fps = 30.0
            
            frame_size = (cv_image.shape[1], cv_image.shape[0]) 
            
            video_out = cv2.VideoWriter(save_path, fourcc, fps, frame_size) 
            
            if not video_out.isOpened():
                rospy.logerr("Failed to open video writer.")
                video_save = False
            else:
                video_save = True
                rospy.loginfo("Video recording started.")
        else:
            if video_out is not None:
                
                video_out.release()
                video_out = None 
                
            
            video_save = False  
            rospy.loginfo("Video recording stopped.")      
            rospy.loginfo("Video saved to {}".format(save_path))
        
        
             
        
        
if __name__ == "__main__":
    
    rospy.init_node("Image_sub")
    
    bridge =CvBridge()
      
    image_sub =rospy.Subscriber("/camera/color/image_raw",Image,ImageCallback,queue_size=10)
    
    rospy.spin()
    
    cv2.destroyAllWindows()