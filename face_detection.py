import cv2
import numpy as np

# Configuracoes

HAAR_CASCADE_PATH = 'models/haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

if face_cascade.empty():
    print(f"ERRO: Nao foi possivel carregar o classificador Haar Cascade em '{HAAR_CASCADE_PATH}'.")
    exit()

# Inicializa a camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERRO: Nao foi possivel abrir a camera.")
    exit()


# Deteccao em tempo real
while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao receber o frame. Saindo...")
        break

    # Converte o frame para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Realiza a detecção facial
    # detectMultiScale(imagem, scaleFactor, minNeighbors)
    #   - scaleFactor: Reduz o tamanho da imagem por este fator em cada escala de imagem.
    #   - minNeighbors: Quantos vizinhos cada retangulo candidato deve ter para ser considerado um rosto.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 

    # Desenha retangulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
        cv2.putText(frame, "Rosto Detectado", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


    # Exibe o frame com as detecoes
    cv2.imshow('Deteccao Facial com Haar Cascade', frame)

    # Sai do loop se 'q' for pressionado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
