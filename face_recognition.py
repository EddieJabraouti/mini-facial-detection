import cv2 

from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True: 

    ret, frame = cap.read()
    if not ret: 
        break 

    try:
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)

        for face in result: 
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            label = f"{face['dominant_emotion']}, {face['gender']}, {int(face['age'])}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    except Exception as e: 
        print("No face Detected or error: ", e)

    cv2.imshow("Deepface Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()