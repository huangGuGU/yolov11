from ultralytics import YOLO
import os
import cv2


def draw_box(image, save, xywh,save_flag):
    x = xywh[0]
    y = xywh[1]
    w = xywh[2]
    h = xywh[3]
    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2

    overlay = image.copy()
    output = image.copy()
    alpha = 0.2
    # 绘制透明填充
    cv2.rectangle(overlay, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), thickness=-1)

    # 叠加透明图层
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # 绘制边框
    cv2.rectangle(output, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), thickness=5)

    # 保存图像
    if save_flag:
        cv2.imwrite(save, output)
    else:
        cv2.imshow('camera_output', output)


def inference(imgdir_path, save_path, weights_path, generate_label):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = YOLO(weights_path)
    try:
        imgdir_path.remove('.DS_Store')
    except:
        pass
    img_list = os.listdir(imgdir_path)
    try:
        img_list.remove('.DS_Store')
    except:
        pass
    for img_name in img_list[::1]:
        if img_name[0] == '.':
            continue
        img_path = os.path.join(imgdir_path, img_name)
        save = os.path.join(save_path, img_name)
        results = model(img_path)
        image = cv2.imread(img_path)
        origin_shape = results[0].orig_shape
        xywh_list = results[0].boxes.xywh.tolist()
        for xywh in xywh_list:
            if generate_label:
                x = xywh[0] / origin_shape[1]
                y = xywh[1] / origin_shape[0]
                w = xywh[2] / origin_shape[1]
                h = xywh[3] / origin_shape[0]

                info = f"0 {x} {y} {w} {h}"
                with open(f"/Users/hzh/Desktop/label/{img_name[:-4]}.txt", "w") as file:
                    file.write(info)
            else:
                draw_box(image, save, xywh,True)


if __name__ == '__main__':
    imgdir_path = r'/Users/hzh/Desktop/catframe'
    save_path = r'/Users/hzh/Desktop/catframe_inferred'
    inference(imgdir_path, save_path, "/Volumes/T7Shield/v11/ultralytics/runs/detect/train34/weights/best.onnx",False)
