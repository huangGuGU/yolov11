import cv2
import os


def frames2video(image_folder, output_video, fps=60):
    try:
        os.listdir(image_folder).remove('.DS_Store')
    except:
        pass
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))],
                    key=lambda x: int(x.split('_')[-1][:-4]))

    if not images:
        print("No images found in the directory!")
        exit()
    # 读取第一张图像以获取视频宽高
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 遍历图像并写入视频
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)

    # 释放资源
    video.release()

    print(f"Video saved as {output_video}")


if __name__ == '__main__':
    image_folder = r'C:\\Users\\49610\\Desktop\\catresultss'
    save_output_video = 'output_video.mp4'
    fps = 60
    frames2video(image_folder, save_output_video, fps)
