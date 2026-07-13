import cv2 as cv
import numpy as np
from deepface import DeepFace

def main():
    proto_path = "models/deploy.prototxt"
    model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

    print("[INFO] Loading face detection model...")
    net = cv.dnn.readNetFromCaffe(proto_path, model_path)

    cap = cv.VideoCapture(0)
    print("[INFO] Starting video camera. Press 'x' to cancel.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Could not grab frame")
            break

        h, w = frame.shape[:2]
        
        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detection = net.forward()

        display_frame = cv.flip(frame, 1)

        for i in range(detection.shape[2]):
            face_confidence = detection[0, 0, i, 2]

            if face_confidence > 0.5:
                box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                
                try:
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        analysis = DeepFace.analyze(
                            img_path=face_roi, 
                            actions=['age', 'emotion'], 
                            enforce_detection=False,
                            silent=True
                        )
                        
                        result = analysis[0]
                        age = int(result['age'])
                        emotion = result['dominant_emotion']
                        
                        emotion_confidence = result['emotion'][emotion]
                        
                        face_str = f"Face: {face_confidence * 100:.1f}%"
                        age_str = f"Age: {age}"
                        emotion_str = f"{emotion.capitalize()}: {emotion_confidence:.1f}%"
                        
                        fx1 = w - x2
                        fx2 = w - x1
                        
                        cv.rectangle(display_frame, (fx1, y1), (fx2, y2), (0, 255, 0), 2)
                        
                        cv.putText(display_frame, face_str, (fx1, y1 - 40 if y1 - 40 > 10 else y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                        cv.putText(display_frame, age_str, (fx1, y1 - 25 if y1 - 25 > 10 else y2 + 35), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                        cv.putText(display_frame, emotion_str, (fx1, y1 - 10 if y1 - 10 > 10 else y2 + 50), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                except Exception as e:
                    pass
        
        cv.imshow("Face Detection", display_frame)
        
        if cv.waitKey(1) & 0xFF == ord('x'):
            break
            
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()