from ultralytics import YOLO
import cv2
from sort import *

cap = cv2.VideoCapture("images/car_fast.mp4")
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolov8l.pt")

classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant',
           'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
           'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
           ]

tracker = Sort(max_age = 3, min_hits = 3, iou_threshold=0.01)
limit = (0, 500, 1280, 500)
total = 0
listID = []

while True:
    success, img = cap.read()
    mask = cv2.imread("images/mask.png")
    masked_img = cv2.bitwise_and(img, mask)
    results = model(masked_img, stream = True)
    
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Confidence
            conf = round(float(box.conf[0]),2)            
            # Class Name
            cls = int(box.cls[0])
            class_name = classes[cls]
            
            if class_name in ["car", "truck", "bus", "motorbike"] and conf > 0.5:
                currentArray = np.array([x1, y1, x2, y2, conf]) 
                detections = np.vstack((detections, currentArray))       
      
    resultsTracker = tracker.update(detections) 
    cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0, 0, 255), 2)
    cv2.putText(img, f"Counter: {total}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness = 2)
     
    for result in resultsTracker:
        x1, y1, x2, y2, id = result            
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color = (255,0,255), thickness=2)
        cv2.putText(img, f"{int(id)}", (max(0,x1), max(35,y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, thickness = 2) 
        
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED) 
        
        if limit[0] < cx < limit[2] and limit[1] - 50 < cy < limit[3] + 50:
            if id not in listID:
                total += 1  
                listID.append(id)    
                cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0, 255, 0), 2)
             
    cv2.imshow("Image", img)
    cv2.waitKey(1)