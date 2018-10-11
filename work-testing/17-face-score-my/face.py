import cv2
import face_recognition
from keras.models import load_model
import numpy as np

model = load_model('faceRank.h5')

def recognition():
    camera = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_DUPLEX

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            x = left
            y = top
            w = right - left
            h = bottom - top
            #image = frame[]

            image = cv2.cvtColor(cv2.resize(gray[y:y + h, x:x + w], (128, 128)), cv2.COLOR_GRAY2BGR)
            image = np.array(image).reshape(1, 128, 128, 3)
            p = model.predict_classes([image])

            cv2.putText(frame, 'score:'+str(p[0]), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow("video", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognition()
