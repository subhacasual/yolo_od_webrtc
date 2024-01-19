from ultralytics import YOLO
import os
import cv2

# Loading the model,.
model = YOLO(
    os.path.join("weights", "yolov8n.pt")
)


def plot_box(img, boxes_list, cls_dict, cls_keys):
    """
    This function plots the bounding boxes on the image and returns the image.

    """
    for ix in range(len(boxes_list)):
        x1, y1, x2, y2 = boxes_list[ix]
        print(x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # adding text to image
        cv2.putText(img, cls_dict[cls_keys[ix]], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return img


# creating function to detect objects and return plotted frame
def detect_objects(frame, model = model):

    # getting the results
    results = model(frame)

    # getting the full class dict
    class_dict = results[0].names # {0: 'person', 1:'byclycle', ...}

    # getting the bounding boxes, xyxy
    boxes = results[0].boxes
    boxes_xyxy = boxes.xyxy.tolist() # [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
    boxes_list = [list(map(int, box)) for box in boxes_xyxy] # Converting to integers.

    # plotting bbox and class names 
    frame = plot_box(frame, boxes_list, class_dict, boxes.cls.tolist())

    return frame
