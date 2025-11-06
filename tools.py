# criar venv (Mac/Linux): python3 -m venv venv
# ativar venv (Mac/Linux): source venv/bin/activate

# criar venv (Windows): python -m venv venv
# ativar venv (Windows): venv\Scripts\activate

# instalar bibliotecas: pip install -r requirements.txt

import cv2
import numpy as np

# Configuracao
IMAGE_PATH = 'images/will.jpg' 
WINDOW_NAME = 'OpenCV Transformacoes Classicas'

img_bgr = cv2.imread(IMAGE_PATH)

if img_bgr is None:
    print(f"ERRO: Nao foi possivel carregar a imagem em '{IMAGE_PATH}'.")

# Exibe a imagem original
cv2.imshow(f"{WINDOW_NAME} - 1. Original (BGR)", img_bgr)

# Transforma em tons de cinza
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
cv2.imshow(f"{WINDOW_NAME} - 2. Tons de Cinza", img_gray)


# Binarizacao usando limiar fixo
limiar, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow(f"{WINDOW_NAME} - 3. Binarizacao (Limiar 127)", img_binary)


# Deteccao de bordas usando o algoritmo Canny
img_edges = cv2.Canny(img_gray, 50, 150) 
cv2.imshow(f"{WINDOW_NAME} - 4. Deteccao de Bordas (Canny)", img_edges)

# Espera ate que uma tecla seja pressionada
cv2.waitKey(0)

# Destroi todas as janelas
cv2.destroyAllWindows()