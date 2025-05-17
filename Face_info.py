import cv2
import insightface
from insightface.app import FaceAnalysis

# Inicializa o modelo InsightFace para análise facial
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # ou GPU se disponível
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 para CPU

cap = cv2.VideoCapture(0)
print("Pressione 'q' para sair...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecta e analisa faces
    faces = app.get(frame)

    for face in faces:
        # Pega bbox da face detectada
        bbox = face.bbox.astype(int)

        # Extrai atributos
        gender = face.gender  # 0: feminino, 1: masculino
        age = face.age

        # Desenha retângulo
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)

        # Exibe idade e gênero
        text = f"Genero: {'M' if gender == 1 else 'F'}, Idade: {int(age)}"
        cv2.putText(frame, text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow('InsightFace Gender & Age', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
