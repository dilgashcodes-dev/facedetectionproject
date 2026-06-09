import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.circle(
            frame,
            (x + w//2, y + h//2),
            w//2,
            (0, 255, 0),
            3
        )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()