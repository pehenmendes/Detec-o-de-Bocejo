from math import dist
import numpy as np
import dlib
import cv2

# definir constantes
MOUTH_AR_THRESH = 0.75
MOUTH_AR_CONSEC_FRAMES = 15
MOUTH_OPEN = False
COUNTER = 0
contagem = 0

def mouth_aspect_ratio(mouth):
    # distâncias verticais
    A = dist(mouth[2], mouth[10])
    B = dist(mouth[4], mouth[8])

    # distância horizontal
    C = dist(mouth[0], mouth[6])

    mar = (A + B) / (2.0 * C)
    return mar

# inicializa o detector e preditor do dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# pega os índices do previsor, para olhos esquerdo e direito
(mStart, mEnd) = (48, 68)

# inicializar vídeo
video_path = 0  # Altere para o caminho do seu vídeo
vs = cv2.VideoCapture(video_path)

# loop sobre os frames do vídeo
while True:
    ret, frame = vs.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectar faces (grayscale)
    rects = detector(gray, 0)

    # loop nas detecções de faces
    for rect in rects:
        
        shape = predictor(gray, rect)
        #devolve shape em uma lista coords
        coords = np.zeros((shape.num_parts, 2), dtype=int)
        for i in range(0, 68): #São 68 landmark em cada face
            coords[i] = (shape.part(i).x, shape.part(i).y)


        # extrair boca
        mouth = coords[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        # desenhar contorno da boca
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # checar ratio x threshold
        if mar > MOUTH_AR_THRESH:
            COUNTER += 1

            if not MOUTH_OPEN:
                contagem += 1
                MOUTH_OPEN = True

            if COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                cv2.putText(frame, "[ALERTA] BOCEJO!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            MOUTH_OPEN = False

        cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Contagem de bocejos: {contagem}", (200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Exibe resultado
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # tecla para sair do script "q"
    if key == ord("q"):
        break

# clean
cv2.destroyAllWindows()
vs.release()
