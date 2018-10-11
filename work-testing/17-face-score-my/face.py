import cv2
import face_recognition
from keras.models import load_model
import numpy as np

score_model = load_model('faceRank.h5')
sex_model = load_model('faceSex.h5')

def recognition():
    camera = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_DUPLEX

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            image = cv2.cvtColor(cv2.resize(gray[top:bottom, left:right], (128, 128)), cv2.COLOR_GRAY2BGR)
            image = np.array(image).reshape(1, 128, 128, 3)

            score = score_model.predict_classes(image)
            sex = sex_model.predict_classes(image)

            if sex == 0:
                sex = 'male'
            else:
                sex = 'female'

            cv2.putText(frame, str(score[0])+' '+sex, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow("video", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognition()
