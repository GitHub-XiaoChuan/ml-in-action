# coding: utf-8
import cv2

datapath = "/Users/xingoo/PycharmProjects/ml-in-action/work-testing/14-face-recognition/img"

def generate():
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

            cv2.imwrite(datapath + '/%s.jpg' % str(count), f)
            count += 1
            print(count)

        cv2.imshow("camera", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate()
