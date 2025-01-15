from ultralytics import YOLO
import os

# 加载模型
model = YOLO(r"../runs/detect/train34/weights/best.pt")

# 在图像上执行对象检测
imgdir_path = r'/Users/hzh/Desktop/cat'
img_list = os.listdir(imgdir_path)
for img_name in img_list[::5]:
    if img_name[0] == '.':
        continue
    img_path = os.path.join(imgdir_path, img_name)
    results = model(img_path)

    origin_shape = results[0].orig_shape

    xywh_list = results[0].boxes.xywh.tolist()
    clazz = results[0].boxes.cls.tolist()
    try:
        cat_index = clazz.index(0)
    except:
        continue

    xywh = xywh_list[cat_index]

    x = xywh[0] / origin_shape[1]
    y = xywh[1] / origin_shape[0]
    w = xywh[2] / origin_shape[1]
    h = xywh[3] / origin_shape[0]

    info = f"0 {x} {y} {w} {h}"
    with open(f"G:\\v11\\label\\{img_name[:-4]}.txt", "w") as file:
        file.write(info)

    print(f"{img_name} done")
