import cv2
import face_recognition
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np

FILE_PATH = "/Users/xingoo/Desktop/sense_go_log.log"
lines = []

with open(FILE_PATH, 'r') as f:
    origin_lines = f.readlines()

for index, line in enumerate(origin_lines):
    jo = json.loads(line.strip('\n'))
    # print(jo)

    f = base64.b64decode(jo['image'])
    image_pil = Image.open(BytesIO(f))
    img = np.array(image_pil.convert('RGB'))

    left = jo['rect']['left']
    top = jo['rect']['top']
    right = jo['rect']['right']
    bottom = jo['rect']['bottom']

    face_image = cv2.resize(img[top:bottom, left:right, :], (128, 128))

    encodings = face_recognition.face_encodings(face_image, num_jitters=10)

    if len(encodings) > 0:
        lines.append(str(6000 + index) + " " + " ".join([str(x) for x in encodings[0]]) + "\n")
        print(lines[-1])

with open("/Users/xingoo/PycharmProjects/ml-in-action/work-testing/17-face-score-my/db2.txt", 'w') as f:
    f.writelines(lines)