# treina e faz uso do modelo para classificar em uma das categorias
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# definindo tamanho padrão das imagens
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
# classificações
CLASS_NAMES = ["glass", "metal", "organic", "paper", "plastic", "others"]

def build_model(num_classes=len(CLASS_NAMES)):
    base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                             include_top=False,
                                             weights='imagenet')
    base.trainable = True

    for layer in base.layers[:-30]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(data_dir, model_out="models/waste_classifier.h5", epochs=30):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.2,
        brightness_range=[0.7, 1.6],
        horizontal_flip=True
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model = build_model(num_classes=train_gen.num_classes)
    model.fit(train_gen, epochs=epochs)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save(model_out)
    print(f"Modelo salvo em {model_out}")
    return model

def load_model(path):
    return tf.keras.models.load_model(path)

#capta caractersticas especficas do vidro 
def analyze_glass_characteristics(img):
    #converte cor para cinza para melhor detecção
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #detecta brilho e reflexos do objeto da imagem (vidro reflete mais luz)
    #calcula quantidade em % de pixels brilhantes na imagem
    _, very_bright = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    very_bright_ratio = np.sum(very_bright) / very_bright.size
    
    _, bright = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    bright_ratio = np.sum(bright) / bright.size
    
    # identifica o padrão de contraste na textura com partes mais escuras e claras
    contrast = gray.std()
    
    # analisa os reflexos/manchas birlhantes na textura do vidro, "conta" quantidade de reflexos
    kernel = np.ones((5,5), np.uint8)
    dilated_bright = cv2.dilate(very_bright, kernel, iterations=1)
    num_bright_clusters = cv2.connectedComponents(dilated_bright)[0] - 1

    #retorno de 4 caracteristicas com valores de carcteristicas de padrão de vidro
    return very_bright_ratio, bright_ratio, contrast, num_bright_clusters

#detecção dos amassados/deformidades no objeto (plastico costuma ser um material mais amassado)
def detect_crushing_deformation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detecta linhas escuras de amassamento e dobras
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    sharp_edges = np.abs(laplacian)
    
    # detecta linhas de vinco/amassados (mudanças bruscas de intensidade)
    _, crease_mask = cv2.threshold(sharp_edges, np.percentile(sharp_edges, 90), 255, cv2.THRESH_BINARY)
    #calcula quantidade em porcentagem da imagem tem esses vincos/amassados
    crease_ratio = np.sum(crease_mask > 0) / crease_mask.size
    
    # analisa as irregularidades na superficie do obejto
    #filtro de medianablur remove ruídos, manchas sujeiras e demais variações
    median = cv2.medianBlur(gray, 5)
    surface_variation = np.abs(gray.astype(float) - median.astype(float))
    #calcula a diferença entre a imagem original e filtrada
    high_variation = np.sum(surface_variation > 15) / surface_variation.size
    
    kernel_size = 7
    #calcula a média local da intensidade dos pixels dentro da janela (suaviza variações pequenas)
    mean_local = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    variance_local = cv2.blur((gray.astype(float) - mean_local)**2, (kernel_size, kernel_size))
    #mede o nivel geral de variação da textura (quanto mais alto, mais "caótica" e amassada está essa superfcie)
    chaos_score = np.std(variance_local)
    
    # detecta bordas fortes na imagem (transições bruscas entre claro e escuro)
    edges = cv2.Canny(gray, 50, 150)
    #encontra linhas retas com base nas bordas detectadas (linhas comuns em vincos de amassamento)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    #conta quantas linhas angulares(linhas de dobras, amassados, quinas e etc) foram detectadas
    num_angular_lines = len(lines) if lines is not None else 0
    
    return crease_ratio, high_variation, chaos_score, num_angular_lines

#detecta caracteristicas do plastico pela textura
def analyze_plastic_characteristics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # analise de textura, plasticos tem uma textura mais aspera
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    #avalia quantidade de variações bruscas na 
    # textura (valor muito alto indica textura mais aspera, provavel plastico)
    texture_roughness = np.var(laplacian)
    
    #canny para detectar bordas do objeto
    edges = cv2.Canny(gray, 30, 100)
    #calcula % de pixels que são bordas no objto
    edge_complexity = np.sum(edges > 0) / edges.size
    
    #gaussianblur blur para reduzir 
    # ruídos/sujeiras e pequenos detalhes que podem ser confundidos como textura
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    detail_loss = np.mean(np.abs(gray.astype(float) - blur.astype(float)))
    
    return texture_roughness, edge_complexity, detail_loss

#analisa propriedades de cor para diferenciar vidro e plastico coloridos
def analyze_color_properties(img):
    #converte cor para HSV para melhor análise de cor (matiz (cor), saturaçao (intensidade), valor (brilho))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # avalia o quão pura é aquela cor
    saturation = hsv[:, :, 1]
    # media da saturação = quão colorida a imagem em geral
    sat_mean = np.mean(saturation)
    # baixo = cor uniforme (vidro colorido)
    # alto = cor varia muito (possivelmente impresso/pintado/rotlos)
    sat_std = np.std(saturation)
    
    # avalia uniformidade da cor
    color_uniformity = 1.0 / (sat_std + 1)  # quanto menor variação, mais uniforme
    
    # avalia intensidade da cor (vidro costuma ter cores mais puras)
    value = hsv[:, :, 2]
    val_mean = np.mean(value)
    
    return sat_mean, sat_std, color_uniformity, val_mean

#analisa padrões de transparencia e translucidez do objeto
def analyze_transparency_and_translucency(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #aplicica blur (desfoque) para suavizar ruidos/sujeiras/detalhes
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # distorção ótica   
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    distortion_score = np.std(laplacian)
    
    #captura somente os pixels muito brilhantes
    _, highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    #avalia variança dos reflexos/brilhos
    # (bordas muito nitidas indicam vidro | reflexos mais suaves indicam plastico)
    highlight_sharpness = cv2.Laplacian(highlights, cv2.CV_64F).var()
    
    # avalia consistencia das superficie
    # vidro tem superfcie mais uniforme, enquanto plastico pode ter mais irregular
    surface_uniformity = cv2.Sobel(gray, cv2.CV_64F, 1, 1).var()
    
    return distortion_score, highlight_sharpness, surface_uniformity

def analyze_surface_smoothness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # analise as micro texturas, plastico costuma tem mais micro texturas
    kernel = np.ones((3,3), np.float32)/9
    #aplica filtro customizado na imagem
    filtered = cv2.filter2D(gray, -1, kernel)
    #compara original com filtrada
    texture_detail = np.mean(np.abs(gray - filtered))
    
    # regularidade de bordas, detecta bordas com o canny
    edges = cv2.Canny(gray, 50, 150)
    edge_smoothness = cv2.blur(edges.astype(float), (5,5)).var()
    
    return texture_detail, edge_smoothness

def analyze_metal_characteristics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # metais têm reflexos mais fortes e concentrados, portando é feito uma análise desse brilho do objeto
    _, very_bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    #calcula em % quanto o objeto é brilhante
    bright_ratio = np.sum(very_bright) / very_bright.size
    
    # canny detecta as bordas do objeto
    #metais em especifico tem bordas muito nitidas (latinhas são cilindricas, com contornos mais definidos)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    
    # superfície lisa com alto contraste de claro e escuro (padrão metálico)
    contrast = gray.std()
    
    # metais refletem luz de forma especular (manchas brilhantes isoladas)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(very_bright, kernel, iterations=1)
    num_reflections = cv2.connectedComponents(dilated)[0] - 1
    
    return bright_ratio, edge_density, contrast, num_reflections

def predict_crop(model, crop_img):
    # redimensiona e normaliza a imagem (224x224 e valores esperados pela rede neural)
    img_processed = tf.image.resize(crop_img, IMG_SIZE)
    img_processed = img_processed / 255.0
    
    # obter predições do modelo
    predictions = model.predict(np.expand_dims(img_processed, 0))[0]
    
    # chama funções que analisam características físicas detalhadas do objeto   
    # (cada função retorna valores numéricos das características)
    distortion, highlights, uniformity = analyze_transparency_and_translucency(crop_img)
    texture, edge_smooth = analyze_surface_smoothness(crop_img)
    bright_ratio, edge_density, contrast, reflections = analyze_metal_characteristics(crop_img)
    
    # metal tem: muito brilho, alto contraste entre claro e escuro, reflexos isolados
    if bright_ratio > 0.15 and contrast > 50 and reflections > 2:
        # muito provável ser metal
        predictions[CLASS_NAMES.index('metal')] += 0.4
        predictions[CLASS_NAMES.index('plastic')] -= 0.2
        predictions[CLASS_NAMES.index('glass')] -= 0.2    
    
    # se o modelo disse "vidro" mas características indicam plástico, corrige!
    if predictions[CLASS_NAMES.index('glass')] > 0.3:
        # caso tenha muita distorçao, muita textura áspera (>0.5), provevelmente é plastico
        if (distortion > 0.8 or texture > 0.5):
            # faz a correção de prefictions, aumenta plástico e diminui vidro
            predictions[CLASS_NAMES.index('plastic')] += 0.3
            predictions[CLASS_NAMES.index('glass')] -= 0.3
    
    # normalizar as probabilidades depois de ajustes
    predictions = np.clip(predictions, 0, 1)  # garante que não fique negativo
    predictions = predictions / np.sum(predictions)
    
    # retorna a categoria e a confiança (exemplo: "metal", 0.85 = 85% de certeza)
    label = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))   

    return label, confidence