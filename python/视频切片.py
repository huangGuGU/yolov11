import cv2
import os


def video2frames(video_path, frames_save_path, skip_frames=1):
    vidcap = cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1]

    if not os.path.exists(frames_save_path):
        os.makedirs(frames_save_path)

    # 计算每隔多少帧保存一次
    frame_interval = max(1, skip_frames)

    # 读取视频中的第一帧图像
    success, image = vidcap.read()

    count = 0
    while success:
        success, image = vidcap.read()

        if count % frame_interval == 0:
            file_path = os.path.join(frames_save_path, f"{video_name}_{count}.jpg")
            if image is not None:
                cv2.imencode('.jpg', image)[1].tofile(file_path)
        count += 1
    vidcap.release()


if __name__ == '__main__':
    path = r'/Users/hzh/Desktop/catvideo'
    frames_save_path = r'/Users/hzh/Desktop/catframe'
    skip_frames = 1
    video2frames(path, frames_save_path, skip_frames)
