import cv2
from deepface import DeepFace

# Inicializa a webcam
cap = cv2.VideoCapture(0)

print("Pressione 'q' para sair...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Análise de rosto
        result = DeepFace.analyze(frame, actions=['gender'], enforce_detection=False)
        gender = result[0]['gender']

        # Adiciona o gênero no frame
        cv2.putText(frame, f"Genero: {gender}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        cv2.putText(frame, f"Erro: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Exibe o frame
    cv2.imshow("Detecção de Gênero", frame)

    # Sai com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
