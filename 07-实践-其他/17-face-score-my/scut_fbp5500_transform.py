import cv2
import face_recognition
import os

IMAGES_PATH = "/Users/xingoo/Documents/dataset/SCUT-FBP5500_v2/Images"
LABLE_FILE_PATH = "/Users/xingoo/Documents/dataset/SCUT-FBP5500_v2/train_test_files/All_labels.txt"
SAVE_PATH  = "/Users/xingoo/Documents/dataset/total_face/"

# 读取评分数据
scores = {}
with open(LABLE_FILE_PATH, 'r') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.strip("\n").split(" ")
        scores[arr[0]] = arr[1]
print(len(scores))

# 截取脸部图像，并重新命名
for index, file_name in enumerate(os.listdir(IMAGES_PATH)):

    print("%d" % index)

    origin_img = cv2.imread(IMAGES_PATH+"/"+file_name)
    face_locations = face_recognition.face_locations(origin_img)

    if len(face_locations) > 0:
        (top, right, bottom, left) = face_locations[0]
        # 截取图片
        face_image = cv2.resize(origin_img[top:bottom, left:right, :], (128, 128))

        # 名称_性别_评分.jpg
        sex = file_name[1]
        name = file_name.split(".")[0][2:]
        score = float(scores[file_name])*2

        save_name = name + "_" + sex + "_" + str(score) + ".jpg"

        cv2.imwrite(SAVE_PATH + save_name, face_image)
