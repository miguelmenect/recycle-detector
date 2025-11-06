#arquivo utils.py, o qual contém os utilitário de todo o projeto
#importando lib cv2
import cv2

#dicionario o qual mapeia objetos detectados, no caso de garrafa(bottle) ele classifca como Plastico
LABEL_TO_CATEGORY = {
    "cup": "Plastico",
    "glassvase": "Vidro",
    "apple": "Organico",
    "banana": "Organico",
    "orange": "Organico",
    "book": "Papel",
    "newspaper": "Papel",
    "can": "Metal",
    "tin": "Metal",
    # para adicionar mais
}

#função de recorte(crop) uma parte da imagem e retorna a mesma ajustada
def crop_bbox(frame, bbox, margin=5):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(w, int(x2) + margin)
    y2 = min(h, int(y2) + margin)
    return frame[y1:y2, x1:x2]

#função que converte objeto detectado para sua respectiva categoria de reciclagem
def map_label_to_category(label):
    return LABEL_TO_CATEGORY.get(label, "Outros") #caso não encontre exibe outros

#adiciona o texto da categoria na parte superior esquerda do frame
def draw_category_text(frame, bbox, category, conf=None):
    x1, y1, x2, y2 = bbox
    text = f"{category}" #text com com categoria
    
    if conf is not None:       # se houver valor de confirmação em %, ele adiciona ao lado da
        text += f" {conf:.2f}" # categoria formatado com 2 casas decimais

    # sesenha o texto na imagem acima do objeto detectado
    # parâmetros: imagem, texto, posição (x,y), fonte, tamanho, cor verde (0,255,0), espessura 2   
    cv2.putText(frame, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    #retorna imagem com texto e de confirmação % de acerto
    return frame
