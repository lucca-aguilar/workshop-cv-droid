import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import time 

MODEL_PATH = "models/best.pt"  
WEBCAM_INDEX = 0

CLASS_NAMES = ["rock", "paper", "scissors"] 


track_history = defaultdict(lambda: [])
seguir = True           
deixar_rastro = False   
window_name = "YOLOv8 RPS - Pedra, Papel, Tesoura"

cap = cv2.VideoCapture(WEBCAM_INDEX)

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"ERRO: Nao foi possivel carregar o modelo em '{MODEL_PATH}'. Verifique o caminho. Detalhe: {e}")
    exit()

if not cap.isOpened():
    print("ERRO: Nao foi possivel abrir a camera. Verifique se o indice esta correto.")
    exit()


cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 

print(f"Webcam iniciada. Pressione 'q' para sair. Classes: {CLASS_NAMES}")
prev_time = time.time() 

while True:
    success, img = cap.read()

    if not success:
        print("Falha na leitura do frame.")
        break

    # 2.1. INFERÊNCIA
    if seguir:
        # Modo de rastreamento: retorna caixas, classes e IDs de rastreamento
        results = model.track(img, persist=True, verbose=False)
    else:
        # Modo de detecção simples
        results = model(img, verbose=False)

    # 2.2. PROCESSAMENTO E DESENHO
    if results and results[0].boxes.data.numel() > 0:
        
        # O .plot() do Ultralytics já desenha as caixas e rótulos
        img = results[0].plot()

    # 2.3. CALCULO E EXIBIÇÃO DE FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 2.4. EXIBIÇÃO FINAL
    cv2.imshow(window_name, img)

    # Condição de saída
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Desligando o sistema de detecção.")
