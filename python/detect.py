import os
from ultralytics import YOLO
from inference import inference, draw_box
from 视频切片 import video2frames
from 合成视频 import frames2video
import cv2
import numpy as np
import mss


class Detect:
    def __init__(self):
        self.video_dir_path = r'/Users/hzh/Desktop/cat_video'
        self.inferred_video_path = r'/Users/hzh/Desktop/inferred'
        self.frames_path = r'/Users/hzh/Desktop/frames'
        self.weights_path = r'../best.pt'
        self.generate_label = False
        self.real_inference = False

    def online_inference_computer_camera(self):
        model = YOLO(self.weights_path)
        camera_cap = cv2.VideoCapture(0)

        if not camera_cap.isOpened():
            print("无法打开摄像头！")
            return

        success, image = camera_cap.read()
        count = 0
        while success:
            success, image = camera_cap.read()
            if image is not None:
                results = model(image)
                xywh_list = results[0].boxes.xywh.tolist()
                for xywh in xywh_list:
                    draw_box(image, 'save', xywh, False)
                # cv2.imshow('camera_output', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1
        camera_cap.release()
        cv2.destroyAllWindows()

    def online_inference_iphone_camera(self):
        model = YOLO(self.weights_path)

        # 初始化屏幕捕获
        with mss.mss() as sct:
            # 定义捕获区域（这里是全屏，也可以指定部分区域

            while True:
                # 捕获屏幕内容
                monitor = {"top": 386, "left": 16, "width": 608 - 16, "height": 719 - 386}
                screenshot = sct.grab(monitor)
                image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
                results = model(image)
                xywh_list = results[0].boxes.xywh.tolist()
                for xywh in xywh_list:
                    draw_box(image, 'save', xywh, False)
                # cv2.imshow('camera_output', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # 释放资源
            cv2.destroyAllWindows()

    def video_inference(self):
        v_list = os.listdir(self.video_dir_path)
        try:
            v_list.remove('.DS_Store')
        except:
            pass
        for v in v_list:
            video_path = os.path.join(self.video_dir_path, v)
            if self.real_inference:
                self.online_inference_computer_camera()

            else:
                frames_save_path = os.path.join(self.frames_path, v)
                video_save_path = os.path.join(self.inferred_video_path, v)
                if not os.path.exists(self.inferred_video_path):
                    os.makedirs(self.inferred_video_path)
                if self.generate_label:
                    video2frames(video_path, frames_save_path, 5)
                    inference(frames_save_path, frames_save_path, self.weights_path, self.generate_label)
                else:
                    video2frames(video_path, frames_save_path, 1)
                    inference(frames_save_path, frames_save_path, self.weights_path, self.generate_label)
                    frames2video(frames_save_path, video_save_path, 30)


if __name__ == '__main__':
    D = Detect()
    # D.online_inference_computer_camera()
