import os
import shutil

imgdir_path = 'G:\\v11\\newimg'
labeldir_path = 'G:\\v11\\label'

train_dir = 'G:\\v11\\dataset\\mycat\\images\\train'
test_dir = 'G:\\v11\\dataset\\mycat\\images\\test'
val_dir = 'G:\\v11\\dataset\\mycat\\images\\val'
train_label_dir = 'G:\\v11\\dataset\\mycat\\labels\\train'
test_label_dir = 'G:\\v11\\dataset\\mycat\\labels\\test'
val_label_dir = 'G:\\v11\\dataset\\mycat\\labels\\val'

dir_list = [train_dir,test_dir,val_dir,train_label_dir,test_label_dir,val_label_dir]
for dir in dir_list:
    if not os.path.exists(dir):
        os.makedirs(dir)


for index,img_name in enumerate(os.listdir(imgdir_path)):
    img_path = os.path.join(imgdir_path,img_name)
    label_path = os.path.join(labeldir_path,img_name[:-3]+'txt')
    if index%10 <7:
        shutil.copy(img_path, os.path.join(train_dir,img_name))
        shutil.copy2(label_path, os.path.join(train_label_dir,img_name[:-3]+'txt'))
        
    
    elif 9>index%10 >=7:
        shutil.copy(img_path, os.path.join(test_dir,img_name))
        shutil.copy2(label_path, os.path.join(test_label_dir,img_name[:-3]+'txt'))
        
    else:
        shutil.copy(img_path, os.path.join(val_dir,img_name))
        shutil.copy2(label_path, os.path.join(val_label_dir,img_name[:-3]+'txt'))
        