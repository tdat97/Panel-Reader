import cv2
import datetime

def get_time_str():
    now = datetime.datetime.now()
    s = f"{now.year:04d}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}{now.second:02d}"
    s += f"_{now.microsecond:06d}"
    return s

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
# webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()

    if status:
        cv2.imshow("test", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        path = f"./{get_time_str()}.jpg"
        cv2.imwrite(path, frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()