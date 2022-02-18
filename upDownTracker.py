import cv2
from tracker import EuclideanTracker

paramRoot = "ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
configRoot = "ssd_mobilenet_v3_large_coco_2020_01_14/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
labelRoot = "ssd_mobilenet_v3_large_coco_2020_01_14/labels.txt"
videoRoot = "test.mp4"

model = cv2.dnn_DetectionModel(paramRoot, configRoot)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

subtracktor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
tracker = EuclideanTracker()

with open(labelRoot, "rt") as l:
    labels = l.read().rstrip("\n").split("\n")

print(labels)

video = cv2.VideoCapture(videoRoot)
# writer = cv2.VideoWriter("final.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (720, 1280))

key = ord("q")

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
up = 0
down = 0
counts = dict()


def linearEq(x):
    return x // 4 + 100

startPts = (0, linearEq(0))
finishPts = (1280, linearEq(1280))

ret = True
while video.isOpened():
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = subtracktor.apply(gray)

    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.4)

    detections = []
    for ind, conf, bbox in zip(classIndex, confidence, bbox):

        x, y, w, h = bbox
        area = w * h
        isBackground = mask[y:y + h, x:x + w].sum() / 255

        if ind != 1 or w > 100 or h > 200 or isBackground < area / 3:
            continue

        detections.append(bbox)

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cx = x + w // 2
        cy = y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if id not in counts:
            counts[id] = cy - linearEq(cx)
        else:
            if counts[id] > 0 and cy - linearEq(cx) < 0:
                down += 1
            elif counts[id] < 0 and cy - linearEq(cx) > 0:
                up += 1
            else:
                del counts[id]

    cv2.line(frame, startPts, finishPts, (255, 0, 0), 1)

    cv2.putText(frame, "up:{}, down:{}".format(up, down), (50, 50), font, font_scale, (0, 255, 0), 2)
    cv2.imshow("test", frame)
    # writer.write(frame)

    if cv2.waitKey(1) == key:
        break
video.release()
# writer.release()
cv2.destroyAllWindows()
