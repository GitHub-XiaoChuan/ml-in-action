import cv2
import face_recognition
import os

TOTAL_PATH = "/Users/xingoo/Documents/dataset/total_face"

lines = []
for index, file_name in enumerate(os.listdir(TOTAL_PATH)):
    face = cv2.imread(TOTAL_PATH+'/'+file_name)
    encodings = face_recognition.face_encodings(face, num_jitters=10)
    if len(encodings) > 0:
        lines.append(str(index) + " " + " ".join([str(x) for x in encodings[0]])+"\n")
        print(index)

print(len(lines))

with open("/Users/xingoo/PycharmProjects/ml-in-action/work-testing/17-face-score-my/db.txt", 'w') as f:
    f.writelines(lines)