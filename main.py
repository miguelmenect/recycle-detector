# arquivo principal do projeto, quem executa o projeto
import os
import cv2
from classifier import load_model, predict_crop

# modo atualmente, mas posso alternar entre "webcam", "imagem" e "video"
MODE = "video"  # coloque "imagem" caso queira classificar por imagem no images_for_analysis 
                 #ou "webcam" caso queira analisar pela camera da maquina.
                 #ou analise por "video" caso queira analisar um video na pasta video_for_analysis

# carrega o classificador treinado
classifier_model = load_model("models/waste_classifier.h5")

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

            if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1) & 0xFF == ord('1')):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif MODE == "imagem": #caso de imagem estatica ele abre a imagem inserida na pasta images for analysis
        # abaixo o caminho da imagem que quero analisar/classificar
        frame = cv2.imread("data\\images_for_analysis\\image.png")
        if frame is None:
            print("Erro: imagem não encontrada!")
            return

        classify_and_show(frame)

        # espera fechar a janela
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif MODE == "video": #caso de video ele abre o video inserido na pasta video_for_analysis
                          #e avalia o objeto do vídeo
        if not os.path.exists("data\\video_for_analysis\\video.mp4"):
            print(f"Erro: Vídeo não encontrado em: data\video_for_analysis\video.mp4") #caso de erro
            return
        process_video("data\\video_for_analysis\\video.mp4")# aqui ele roda o video    

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

    # desenha o texto na imagem na posição (10, 30)
    # parametros: imagem, texto, posição, fonte, tamanho 1, cor verde, espessura 2, tipo de linha
    text = f"{label} ({confidence*100:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # tittulo do frame da imagem
    cv2.imshow("Descrição de lixo", frame)

if __name__ == "__main__":
    main()
