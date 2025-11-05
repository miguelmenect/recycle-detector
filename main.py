# arquivo principal do projeto, quem executa o projeto
import os
import cv2
import numpy as np
from classifier import load_model, predict_crop

# modo atualmente, mas posso alternar entre "webcam", "imagem" e "video"
MODE = "imagem"  # coloque "imagem" caso queira classificar por imagem no images_for_analysis 
                 #ou "webcam" caso queira analisar pela camera da maquina.
                 #ou analise por "video" caso queira analisar um video na pasta video_for_analysis

key = cv2.waitKey(1) #botão pressionado

# carrega o classificador treinado
classifier_model = load_model("models/waste_classifier.h5")

#array categorizado com informações extras educativas sobre cada tipo de lixo/objeto analizado
EDUCATIONAL_INFO = {
    "pet_bottle": {
        "title": "♻️ Garrafa PET",
        "text": [
            "As garrafas PET são altamente recicláveis e podem ganhar novas",
            "finalidades após o descarte correto. Quando recicladas, elas podem",
            "ser transformadas em fibras têxteis para roupas, enchimento de",
            "almofadas, carpetes e até novas embalagens. Além de reduzir o volume",
            "de lixo nos aterros, a reciclagem do PET diminui o consumo de",
            "petróleo, um recurso não renovável."
        ]
    },
    "plastic": {
        "title": "♻️ Plásticos diversos",
        "text": [
            "Plásticos, quando reciclados, podem ser reaproveitados na",
            "fabricação de brinquedos, caixas organizadoras, baldes, sacolas",
            "recicladas e até modulados para construção. A correta destinação",
            "contribui para evitar poluição dos oceanos, rios e solos,",
            "preservando a vida animal e vegetal."
        ]
    },
    "aluminum_can": {
        "title": "♻️ Latinhas de alumínio",
        "text": [
            "As latinhas de alumínio possuem uma das reciclagens mais eficientes",
            "do mundo. Elas podem retornar para a indústria e se transformarem",
            "novamente em novas latas em poucas semanas. Além de economizar",
            "energia na produção, a reciclagem do alumínio reduz drasticamente",
            "a extração mineral e impactos ambientais."
        ]
    },
    "metal_sheet": {
        "title": "♻️ Chapas de metal",
        "text": [
            "Chapas de metal descartadas podem ser reaproveitadas na fabricação",
            "de peças industriais, estruturas metálicas, utensílios e ferramentas.",
            "O processo de reciclagem desses materiais reduz a necessidade de",
            "extração de minério e aproveita o metal já existente, preservando",
            "recursos naturais."
        ]
    },
    "metal": {
        "title": "♻️ Metais diversos",
        "text": [
            "Metais, como cobre, ferro e aço, podem ser derretidos e",
            "reutilizados infinitamente sem perder qualidade. Com a reciclagem,",
            "é possível criar cabos elétricos, motores, estruturas, arames e",
            "outros componentes industriais, além de reduzir o consumo de energia",
            "quando comparado ao processo de mineração."
        ]
    },
    "paper": {
        "title": "♻️ Papel",
        "text": [
            "O papel reciclado pode se transformar em cadernos, caixas, papel",
            "higiênico, jornais e até artefatos decorativos. Reciclar papel reduz",
            "o desmatamento, o consumo de água e o uso de produtos químicos na",
            "produção. Quando destinado corretamente, contribui para a preservação",
            "de florestas e equilíbrio ambiental."
        ]
    },
    "glass_bottle": {
        "title": "♻️ Garrafas de vidro",
        "text": [
            "As garrafas de vidro podem ser totalmente reaproveitadas e retornarem",
            "ao ciclo como novas garrafas, potes e recipientes. Sua reciclagem",
            "economiza energia industrial e evita o descarte em aterros, já que o",
            "vidro demora séculos para se decompor. Além disso, o vidro pode ser",
            "reciclado indefinidamente sem perder qualidade."
        ]
    },
    "flat_glass": {
        "title": "♻️ Vidros planos",
        "text": [
            "Vidros planos, como os encontrados em janelas e portas, podem ser",
            "reciclados em novos painéis, utensílios, mosaicos, decoração ou",
            "triturados para criação de pisos e revestimentos. A reciclagem deste",
            "material evita acidentes, reduz o descarte inadequado e aproveita um",
            "recurso que leva centenas de anos para se decompor."
        ]
    },
    "glass": {
        "title": "♻️ Vidros diversos",
        "text": [
            "Vidro, quando encaminhados para recicladores especializados,",
            "podem ser derretidos e moldados novamente para inúmeros produtos.",
            "O reaproveitamento impede que fragmentos perigosos prejudiquem o meio",
            "ambiente e evita excessos nos aterros sanitários."
        ]
    },
    "organic": {
        "title": "♻️ Orgânico",
        "text": [
            "Resíduos orgânicos, apesar de não serem recicláveis industrialmente,",
            "possuem grande valor ambiental. Podem ser usados na compostagem,",
            "gerando adubo natural rico em nutrientes para hortas, jardins e",
            "plantações. Além de reduzir mau cheiro e proliferação de vetores,",
            "a compostagem diminui o volume de lixo enviado aos aterros e",
            "contribui para um ciclo sustentável."
        ]
    }
}

def analyze_object_shape(frame):
    # converte para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # aplica threshold para segmentar o objeto
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # encontra contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # pega o maior contorno pois provavelmente é o objeto principal
    largest_contour = max(contours, key=cv2.contourArea)
    
    # calcula caracteristicas geométricas
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # retangulo delimitador
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # aqui ele calcula a circularidade, para saber se o objeto é mais redondo ou não
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # aqui ele calcula o qual alondo é o objeto
    rect = cv2.minAreaRect(largest_contour)
    box_width, box_height = rect[1]
    if box_height > box_width:
        box_width, box_height = box_height, box_width
    elongation = box_width / box_height if box_height > 0 else 0
    
    return {
        'aspect_ratio': aspect_ratio,
        'circularity': circularity,
        'elongation': elongation,
        'area': area
    }

def detect_specific_object(label, frame):    
    # analisa a forma do objeto da imagem
    shape_info = analyze_object_shape(frame)
    
    if shape_info is None:
        # se não conseguiu analisar forma, retorna categoria generica/diversa
        return label
    
    # caso de plastico
    if label == "plastic":
        # garrafa pet: alta proporção altura/largura (>2.0) e um pouquinho cilindrica
        if shape_info['aspect_ratio'] > 2.0 or shape_info['elongation'] > 2.0:
            return "pet_bottle"
        else:
            return "plastic"
    
    # caso de metal
    elif label == "metal":
        # latinhas e semelhates: forma cilindrica (elongation ~1.5-2.5) e circularidade alta
        if 1.3 < shape_info['elongation'] < 3.0 and shape_info['circularity'] > 0.6:
            return "aluminum_can"
        # chapas de metal: forma mais plana e retangular, com finura
        elif shape_info['aspect_ratio'] < 1.5 and shape_info['circularity'] < 0.4:
            return "metal_sheet"
        else:
            return "metal"
    
    # caso de vidro
    elif label == "glass":
        # garrafa de vidro: alta proporção altura/largura
        if shape_info['aspect_ratio'] > 2.0 or shape_info['elongation'] > 2.0:
            return "glass_bottle"
        # vidro plano: forma mais retangular
        elif shape_info['aspect_ratio'] < 1.5 and shape_info['circularity'] < 0.5:
            return "flat_glass"
        else:
            return "glass"
    
    # papel e orgânico não precisam de detecção específica
    else:
        # categorias que não precisam detecção especifica
        return label

def draw_info_box(frame, info_key):
    #objeto retangular com informações educativas sobre o objeto classificado
    if info_key not in EDUCATIONAL_INFO:
        return
    
    info = EDUCATIONAL_INFO[info_key]
    h, w = frame.shape[:2]
    
    # configurações da caixa de texto
    margin = 10
    line_height = 20
    padding = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1
    
    # calcula altura da caixa
    num_lines = len(info["text"]) + 2  # +2 para título e linha vazia
    box_height = num_lines * line_height + 2 * padding
    box_width = w - 2 * margin
    
    # posiçiona a caixa
    box_x1 = margin
    box_y1 = h - box_height - margin
    box_x2 = box_x1 + box_width
    box_y2 = h - margin
    
    # desenha fundo preto quase transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # desenha borda
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
    
    # desenha título (em verde)
    y_offset = box_y1 + padding + line_height
    cv2.putText(frame, info["title"], (box_x1 + padding, y_offset),
                font, font_scale + 0.1, (0, 255, 0), font_thickness + 1, cv2.LINE_AA)
    
    # pula uma linha
    y_offset += line_height + 5
    
    # escreve o texto em branco
    for line in info["text"]:
        cv2.putText(frame, line, (box_x1 + padding, y_offset),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        y_offset += line_height

def main():
    if MODE == "webcam": # se for webcam ele abre a camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro: webcam não encontrada!") #caso de erro.
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            classify_and_show(frame)
            
            if key == ord('q') or key == 27:  #se "q" ou "esc" for pressionado encerra
                cv2.destroyAllWindows()
                break #break para sair do loop  de camera

        cap.release()
        cv2.destroyAllWindows()

    elif MODE == "imagem": #caso de imagem estatica ele abre a imagem inserida na pasta images for analysis
        # abaixo o caminho da imagem que quero analisar/classificar
        frame = cv2.imread("data\\images_for_analysis\\image5.png")
        if frame is None:
            print("Erro: imagem não encontrada!")
            return

        classify_and_show(frame)

        if key == ord('q') or key == 27:  #se "q" ou "esc" for pressionado encerra
            cv2.destroyAllWindows()
            return #retorna nada

        # espera fechar a janela
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif MODE == "video": #caso de video ele abre o video inserido na pasta video_for_analysis
                          #e avalia o objeto do vídeo
        if not os.path.exists("data\\video_for_analysis\\video.mp4"):
            print(f"Erro: Vídeo não encontrado em: data\video_for_analysis\video.mp4") #caso de erro
            return
        process_video("data\\video_for_analysis\\video.mp4")# aqui ele roda o video   
         
        if key == ord('q') or key == 27:  #se "q" ou "esc" for pressionado encerra
            cv2.destroyAllWindows()
            return #retorna nada
        
def process_video(video_path):
    #abre o vídeo    
    cap = cv2.VideoCapture(video_path)

    # verifica se conseguiu abrir o vídeo
    # caso der errado (arquivo não existe, corrompido, etc), para aqui
    if not cap.isOpened():
        print(f"Erro: não foi possível abrir o vídeo {video_path}")
        return
    
    print(f"Processando vídeo: {video_path}")
    print("Pressione 'q' para sair")
    
    #loop que vai processar o vídeo (frame por frame)
    while True:
        ret, frame = cap.read()
        # se não conseguiu ler (ret em false) significa que o vídeo acabou
        if not ret:
            print("Fim do vídeo")
            break
        #o frame atual é analisado e classificado pela função classify_and_show
        classify_and_show(frame)
        #se "q" ou ""1" for pressionado sai do loop e encerra
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

##classifica imagem e mostra o resultado na tela
def classify_and_show(frame):
    # ele envia a imagem para o modelo classificar
    # e retorna: label (categoria) e confirmação (confiança de 0 a 1)
    label, confidence = predict_crop(classifier_model, frame)
    print(f"Categoria: {label}, Confirmação: {confidence*100:.2f}%")

    #identifica o objeto específico baseado na categoria e na forma
    specific_object = detect_specific_object(label, frame)
    print(f"Objeto analisado é: {specific_object}")

    # desenha o texto na imagem na posição (10, 30)
    # parametros: imagem, texto, posição, fonte, tamanho 1, cor verde, espessura 2, tipo de linha
    text = f"{label} ({confidence*100:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # desenha caixa de informação educativa (baseada no objeto específico)
    draw_info_box(frame, specific_object)

    # tittulo do frame da imagem
    cv2.imshow("Descrição de lixo", frame)

if __name__ == "__main__":
    main()
