import cv2
import insightface
from insightface.app import FaceAnalysis
import os
import random
import time

# Resolução vertical para tablet (modo retrato)
largura_tela = 1080
altura_tela = 1920

# Carrega o modelo de análise facial
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Caminhos para imagens
caminho_generico = os.path.join("data", "generic")
caminho_homem = os.path.join("data", "man")
caminho_mulher = os.path.join("data", "woman")

# Função para escolher e redimensionar imagem
def escolher_imagem(pasta):
    imagens = [img for img in os.listdir(pasta) if img.endswith((".png", ".jpg"))]
    if imagens:
        caminho = os.path.join(pasta, random.choice(imagens))
        img = cv2.imread(caminho)
        if img is not None:
            return cv2.resize(img, (largura_tela, altura_tela))
    return None

# Transição suave entre imagens
def transicao_suave(img1, img2, titulo="Vitrine 1", steps=15, delay=30):
    if img1 is None:
        img1 = img2
    for i in range(steps + 1):
        alpha = i / steps
        blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        cv2.imshow(titulo, blended)
        cv2.waitKey(delay)

# Inicializa webcam
cap = cv2.VideoCapture(0)

# Cria janelas
cv2.namedWindow("Vitrine 1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vitrine 1", largura_tela, altura_tela)

cv2.namedWindow("Detecção", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detecção", 640, 480)

cv2.namedWindow("Vitrine 2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vitrine 2", largura_tela, altura_tela)

# Inicializa propaganda
prop_atual = escolher_imagem(caminho_generico)

# Controle de tempo
tempo_final_personalizada = 0
tempo_exibicao_personalizada = 3
tempo_troca_generica = 2
ultimo_tempo_troca_generica = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar da webcam.")
        break

    agora = time.time()
    nova_propaganda = None
    frame_mostrar = frame.copy()

    # Se passou o tempo da propaganda personalizada
    if agora > tempo_final_personalizada:
        faces = app.get(frame)

        if faces:
            face = faces[0]
            gender = face.gender

            # Escolhe imagem personalizada
            pasta_escolhida = caminho_homem if gender == 1 else caminho_mulher
            nova_propaganda = escolher_imagem(pasta_escolhida)
            tempo_final_personalizada = agora + tempo_exibicao_personalizada
        else:
            # Troca genérica
            if agora - ultimo_tempo_troca_generica >= tempo_troca_generica:
                nova_propaganda = escolher_imagem(caminho_generico)
                ultimo_tempo_troca_generica = agora

    # Se a propaganda mudou, faz transição
    if nova_propaganda is not None and nova_propaganda is not prop_atual:
        transicao_suave(prop_atual, nova_propaganda, titulo="Vitrine 1")
        prop_atual = nova_propaganda
    else:
        cv2.imshow("Vitrine 1", prop_atual)

    # Mostra a detecção da câmera (sem desenhar ou mostrar dados)
    cv2.imshow("Detecção", frame_mostrar)

    # Mostra a vitrine duplicada
    cv2.imshow("Vitrine 2", prop_atual)

    # Encerra com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
