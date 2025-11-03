# test_classifier.py - Arquivo para testar o modelo treinado
# importa as funções do arquivo classifier.py
from classifier import load_model, predict_crop
import cv2  # Importa a biblioteca OpenCV

# carrega o modelo treinado do arquivo waste_classifier.h5
model = load_model("models/waste_classifier.h5")

# carrega a imagem de teste
# não use imagens da pasta "train"! Use imagens novas!
img = cv2.imread("data/images_for_analysis/image.jpg")  # Troque para uma imagem de TESTE

# verifica se a imagem foi carregada corretamente
if img is None:
    print("erro: imagem não encontrada!")  # Exibe erro no console
else:
    # classifica a imagem usando o modelo
    label, confidence = predict_crop(model, img)
    
    # mostra o resultado
    print(f'Categoria: {label}, Confirmação: {confidence*100:.2f}%')
                                               #↑ multiplica por 100
