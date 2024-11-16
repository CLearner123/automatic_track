#!/usr/bin/env python3

#该文件用于截取视频流中的文件


import cv2

# 视频文件路径，可以替换为你自己的视频文件路径
video_path = "/home/mei24/catkin_ws/src/automatic_track/video/video_1730468621.8811817.mp4"  # 替换为你的视频路径
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

frame_count = 0  # 用于命名保存的图像文件

while True:
    ret, frame = cap.read()  # 读取视频帧
    if not ret:
        print("Reached the end of the video or cannot read the frame.")
        break

    cv2.imshow("Video Frame", frame)  # 显示当前帧

    key = cv2.waitKey(1)  # 等待键盘输入，1毫秒延时
    if key == ord('s'):  # 按下 's' 键保存当前帧
        frame_filename = f"/home/mei24/catkin_ws/src/automatic_track/images/saved_frame_{frame_count}.jpg"
        cv2.imwrite(frame_filename, frame)  # 保存帧为图像文件
        print(f"Saved frame: {frame_filename}")
        frame_count += 1
    elif key == ord('q'):  # 按下 'q' 键退出
        break

# 释放视频对象和关闭所有窗口
cap.release()
cv2.destroyAllWindows()
