# arquivo principal do projeto, quem executa o projeto
import cv2
from classifier import load_model, predict_crop

# modo atualmente e em imagem estatica, mas posso mudar, "webcam" ou "imagem"
MODE = "imagem"  # troque para "webcam" para usar a câmera do computador

# carrega o classificador treinado
classifier_model = load_model("models/waste_classifier.h5")

def main():
    if MODE == "webcam": # se for webcam ele abre a camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro: webcam não encontrada!") #caso de erro
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            classify_and_show(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif MODE == "imagem": #caso de imagem estatica ele abre a imagem inserida na pasta images for analysis
        # abaixo o caminho da imagem que quero analisar/classificar
        frame = cv2.imread("data\\images_for_analysis\\image.jpg")
        if frame is None:
            print("Erro: imagem não encontrada!")
            return

        classify_and_show(frame)

        # espera fechar a janela
        cv2.waitKey(0)
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
