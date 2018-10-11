import cv2
import os

IMAGE_PATH = "/Users/xingoo/PycharmProjects/ml-in-action/work-testing/17-face-score-my/data/"

for file_name in os.listdir(IMAGE_PATH+"train"):
    arr = file_name.strip("\n").split(".")[0].split("-")
    name = str(int(arr[1])+6000) + "_F_" + arr[0] + ".jpg"
    os.rename(IMAGE_PATH+"train/"+file_name, IMAGE_PATH+"train1/"+name)