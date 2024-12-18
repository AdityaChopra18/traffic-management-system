import cv2
import numpy as np

# Load YOLO
def load_yolo():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Load COCO class labels
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes, output_layers

# Process video and classify vehicles
def process_video(video_path):
    net, classes, output_layers = load_yolo()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        height, width, channels = frame.shape

        # Detect objects using YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Initialize detection results
        class_ids = []
        confidences = []
        boxes = []

        # Parse detection results
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Consider only relevant classes (cars, buses, trucks, motorcycles)
                if confidence > 0.5 and classes[class_id] in ["car", "bus", "truck", "motorbike"]:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression to remove duplicates
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                confidence = round(confidences[i], 2)

                # Draw bounding box and label
                color = (0, 255, 0)  # Green for vehicles
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence}", (x, y - 10), font, 1, color, 1)

        # Display the frame
        cv2.imshow("Vehicle Classification", frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    video_path = input("Enter the path to the traffic video file: ")
    process_video(video_path)
