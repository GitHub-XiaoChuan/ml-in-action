import cv2
import face_recognition


def recognition():
    camera = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_DUPLEX

    while True:
        ret, frame = camera.read()

        face_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, 'score:99999', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow("video", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognition()
