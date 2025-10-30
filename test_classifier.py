# test_classifier.py - Arquivo para testar o modelo treinado
# Importa as funções do arquivo classifier.py
from classifier import load_model, predict_crop
import cv2  # Importa a biblioteca OpenCV

# Carrega o modelo treinado do arquivo waste_classifier.h5
model = load_model("models/waste_classifier.h5")

# Carrega a imagem de teste
# ⚠️ CUIDADO: Não use imagens da pasta "train"! Use imagens novas!
img = cv2.imread("data/images_for_analysis/image.png")  # Troque para uma imagem de TESTE

# Verifica se a imagem foi carregada corretamente
if img is None:
    print("❌ Erro: imagem não encontrada!")  # Exibe erro no console
else:
    # Classifica a imagem usando o modelo
    label, confidence = predict_crop(model, img)
    
    # Mostra o resultado (CORRIGIDO!)
    print(f'Categoria: {label}, Confirmação: {confidence*100:.2f}%')
    #                                           ↑ Multiplica por 100!
    
    # Ou use essa versão (mais simples):
    # print(f'Categoria: {label}, Confiança: {confidence:.2f}')