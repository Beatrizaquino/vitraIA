import cv2
import insightface
from insightface.app import FaceAnalysis
import os
import random
import time

# Carregando o modelo de análise facial
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Definindo os caminhos das imagens que quero usar
caminho_generico = os.path.join("data", "generic")
caminho_homem = os.path.join("data", "man")
caminho_mulher = os.path.join("data", "woman")

# Pega uma imagem aleatória de uma pasta e redimensiona pro padrão do "celular"
def escolher_imagem(pasta):
    imagens = [img for img in os.listdir(pasta) if img.endswith((".png", ".jpg"))]
    if imagens:
        caminho = os.path.join(pasta, random.choice(imagens))
        img = cv2.imread(caminho)
        if img is not None:
            return cv2.resize(img, (480, 800))
    return None

# Faz uma transição entre duas imagens com um efeito suave
def transicao_suave(img1, img2, titulo="Vitro IA", steps=15, delay=30):
    if img1 is None:
        img1 = img2
    for i in range(steps + 1):
        alpha = i / steps
        blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        cv2.imshow(titulo, blended)
        cv2.waitKey(delay)

# Ligando a webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Vitro IA", cv2.WINDOW_NORMAL)

# Começa mostrando uma propaganda genérica até alguém aparecer
prop_atual = escolher_imagem(caminho_generico)

# Controle de tempo pra saber quando trocar as imagens
tempo_final_personalizada = 0
tempo_exibicao_personalizada = 7  # quanto tempo fica a imagem personalizada
tempo_troca_generica = 5  # tempo entre trocas das genéricas
ultimo_tempo_troca_generica = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar da webcam.")
        break

    agora = time.time()
    nova_propaganda = None

    # Só troca se o tempo da imagem personalizada tiver acabado
    if agora > tempo_final_personalizada:
        faces = app.get(frame)

        if faces:
            # Pega a primeira face detectada
            face = faces[0]
            gender = face.gender
            age = int(face.age)

            # Decide qual imagem mostrar com base no gênero
            pasta_escolhida = caminho_homem if gender == 1 else caminho_mulher
            nova_propaganda = escolher_imagem(pasta_escolhida)

            # Marca o tempo até quando essa imagem vai ficar
            tempo_final_personalizada = agora + tempo_exibicao_personalizada
        else:
            # Se não tiver ninguém, vai trocando as genéricas de tempos em tempos
            if agora - ultimo_tempo_troca_generica >= tempo_troca_generica:
                nova_propaganda = escolher_imagem(caminho_generico)
                ultimo_tempo_troca_generica = agora

    # Se a imagem for diferente da atual, faz a transição
    if nova_propaganda is not None and nova_propaganda is not prop_atual:
        transicao_suave(prop_atual, nova_propaganda)
        prop_atual = nova_propaganda
    else:
        # Se nada mudou, só continua exibindo
        cv2.imshow("Vitro Ai", prop_atual)

    # Encerra se apertar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
