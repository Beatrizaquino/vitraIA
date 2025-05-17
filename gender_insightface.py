import cv2
import insightface
from insightface.app import FaceAnalysis

# Inicializa o modelo InsightFace
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Carrega as imagens de propaganda
prop_masculina = cv2.imread('data/man/3.jpg')
prop_feminina = cv2.imread('data/woman/3.jpg')  # substitua se tiver outro caminho

# Verifica se as imagens foram carregadas corretamente
if prop_masculina is None:
    raise FileNotFoundError("Imagem de propaganda masculina não encontrada em 'data/man/3.jpg'")
if prop_feminina is None:
    raise FileNotFoundError("Imagem de propaganda feminina não encontrada em 'data/woman/3.jpg'")

# Redimensiona as propagandas para exibição no canto
prop_masculina = cv2.resize(prop_masculina, (150, 150))
prop_feminina = cv2.resize(prop_feminina, (150, 150))

# Inicia a webcam
cap = cv2.VideoCapture(0)
print("Pressione 'q' para sair...")

frame_count = 0
process_every_n_frames = 5
detected_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % process_every_n_frames == 0:
        detected_faces = app.get(frame)

    propaganda_img = None

    for face in detected_faces:
        bbox = face.bbox.astype(int)
        gender = face.gender
        age = face.age

        # Escolhe a propaganda com base no gênero
        propaganda_img = prop_masculina if gender == 1 else prop_feminina

        # Mostra a caixa e o texto
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        text = f"Genero: {'M' if gender == 1 else 'F'}, Idade: {int(age)}"
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Exibe propaganda no canto inferior direito
    if propaganda_img is not None:
        h, w, _ = propaganda_img.shape
        frame[-h:, -w:] = propaganda_img

    cv2.imshow('InsightFace com Propaganda por Genero', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
