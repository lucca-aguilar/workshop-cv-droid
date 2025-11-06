import cv2
import time
from ultralytics import YOLO

WINDOW_NAME = 'YOLOv8 Detecção em Tempo Real'

model = YOLO('models/best.pt')

class_names = model.names 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERRO: Nao foi possivel abrir a camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao receber o frame. Saindo...")
        break

    results = model(frame, stream=True, verbose=False) 
    
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0].tolist()] 

            confidence = round(box.conf[0].item(), 2)
            class_id = int(box.cls[0].item())       
            class_name = class_names[class_id]       


            cor = (0, 255, 0) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

            label = f'{class_name}: {confidence}'
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detecção encerrada.")