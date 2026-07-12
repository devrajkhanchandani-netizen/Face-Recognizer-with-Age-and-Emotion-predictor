import cv2 as cv
import numpy as np

def main():
    proto_path = "models/deploy.prototxt"
    model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

    print("[INFO] Loading face detection model...")
    net = cv.dnn.readNetFromCaffe(proto_path, model_path)

    cap = cv.VideoCapture(0)
    print("[INFO] Starting video camera. Press X to cancel.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Could not grab frame")
            break

        # 1. Flip the raw frame right away
        frame = cv.flip(frame, 1)

        h, w = frame.shape[:2]

        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detection = net.forward()

        for i in range(detection.shape[2]):
            con = detection[0, 0, i, 2]

            if con > 0.5:
                box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                
                # Everything drawn here is built straight onto the flipped frame
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{con * 100:.1f}%"
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv.putText(frame, label, (x1, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        
        cv.imshow("Face Detection", frame)
        
        if cv.waitKey(1) & 0xFF == ord('x'):
            break
            
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()