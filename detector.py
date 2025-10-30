# detecta os objetos na imagem usando modelos de detecção yolo
import cv2
import cvlib as cv
#importa função que desenha quadrado em torno do objeto detectado
from cvlib.object_detection import draw_bbox

#essa função detecta objetos na imagem usando o modelo yolo
def detect_and_draw(frame, classes_of_interest=None):
    """
    Detect objects in frame using cvlib (YOLO).
    Returns: annotated_frame, detections_list (list of dicts)
    """

    # deetecta objetos na imagem
    # bbox = lista de coordenadas dos retângulos/quadrados ao redor dos objetos
    # labels = lista com os nomes dos objetos detectados
    # confidences = lista com a confiança (0 a 1) de cada detecção
    # confidence=0.35 só aceita detecções com 35% ou mais de confiança
    # model='yolov4' define qual versão do YOLO usar (pode ser yolov3 também)
    bbox, labels, confidences = cv.detect_common_objects(frame, confidence=0.35, model='yolov4')  # ou yolov3
    # cria uma copia da imagem e desenha os retangulos e textos sobre os objetos detectados
    annotated = draw_bbox(frame.copy(), bbox, labels, confidences, write_conf=True)
    #lista vazia que vai armazenar as detecções processadas
    detections = []
    # loop que irá percorrer cada objeto detectado (bbox, label e confirmação juntos)
    for b, lab, conf in zip(bbox, labels, confidences):
        x1, y1, x2, y2 = b #cordenadas do retangulo/quadrado
        if classes_of_interest and lab not in classes_of_interest:
            continue
        #diciona as informações da detecção em formato de dicionario na lista (.append)
        detections.append({
            "bbox": (x1, y1, x2, y2), #cordenadas do retangulo/quadrado na imagem
            "label": lab, #tipo do objeto detectado
            "conf": float(conf) #valor de confirmação da afirmação daquele objeto/lixo
        })
    #retorna a imagem anotada/formatada e a lista de detecções
    return annotated, detections
