
import os
import shutil


# 复制文件


labeldir_path= r'/Users/hzh/Desktop/label'
imgdir_path= r'/Users/hzh/Desktop/frames/IMG_5360.mov'
deletlabelpath = r'/Users/hzh/Desktop/deletelabel'
newimgpath = r'/Users/hzh/Desktop/frames/new+IMG_5360.mov'
if not os.path.exists(deletlabelpath):
    os.mkdir(deletlabelpath)
if not os.path.exists(newimgpath):
    os.mkdir(newimgpath)
label_list = os.listdir(labeldir_path)
try:
    label_list.remove('.DS_Store')
except:
    pass
for label_name in label_list:
    if label_name =="classes.txt":
        continue
    label_path = os.path.join(labeldir_path, label_name)
    delete_label_path = os.path.join(deletlabelpath, label_name)
    with open(label_path, "r", encoding="utf-8") as file:
        content = file.read().strip()  # 去掉多余空白字符

    if not content:
        shutil.move(label_path, delete_label_path)
    else:
        img_path = os.path.join(imgdir_path, label_name[:-3] + 'jpg')
        img_copy_path = os.path.join(newimgpath, label_name[:-3] + 'jpg')
        shutil.copy(img_path, img_copy_path)



