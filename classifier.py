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

def analyze_glass_characteristics(img):
    """Detecta características específicas de vidro"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. BRILHO INTENSO - vidro reflete muito mais luz
    _, very_bright = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    very_bright_ratio = np.sum(very_bright) / very_bright.size
    
    _, bright = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    bright_ratio = np.sum(bright) / bright.size
    
    # 2. CONTRASTE ALTO - vidro tem áreas muito claras e escuras
    contrast = gray.std()
    
    # 3. REFLEXOS ESPECULARES (pontos muito brilhantes concentrados)
    kernel = np.ones((5,5), np.uint8)
    dilated_bright = cv2.dilate(very_bright, kernel, iterations=1)
    num_bright_clusters = cv2.connectedComponents(dilated_bright)[0] - 1
    
    return very_bright_ratio, bright_ratio, contrast, num_bright_clusters

def detect_crushing_deformation(img):
    """Detecta se há amassamento - CRUCIAL: vidro não amassa, só plástico!"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. VINCOS/DOBRAS - linhas escuras de amassamento
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    sharp_edges = np.abs(laplacian)
    
    # Detecta linhas de vinco (mudanças bruscas de intensidade)
    _, crease_mask = cv2.threshold(sharp_edges, np.percentile(sharp_edges, 90), 255, cv2.THRESH_BINARY)
    crease_ratio = np.sum(crease_mask > 0) / crease_mask.size
    
    # 2. IRREGULARIDADES NA SUPERFÍCIE - amassamento cria padrões caóticos
    # Usa filtro de mediana para detectar variações locais
    median = cv2.medianBlur(gray, 5)
    surface_variation = np.abs(gray.astype(float) - median.astype(float))
    high_variation = np.sum(surface_variation > 15) / surface_variation.size
    
    # 3. TEXTURA CAÓTICA - amassamento cria padrão não-uniforme
    # Compara variância em pequenas janelas
    kernel_size = 7
    mean_local = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    variance_local = cv2.blur((gray.astype(float) - mean_local)**2, (kernel_size, kernel_size))
    chaos_score = np.std(variance_local)
    
    # 4. DETECÇÃO DE LINHAS ANGULARES (vincos formam ângulos)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    num_angular_lines = len(lines) if lines is not None else 0
    
    return crease_ratio, high_variation, chaos_score, num_angular_lines

def analyze_plastic_characteristics(img):
    """Detecta características específicas de plástico"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. TEXTURA RUGOSA - plástico amassa, vidro não
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_roughness = np.var(laplacian)
    
    # 2. BORDAS IRREGULARES - plástico deforma mais
    edges = cv2.Canny(gray, 30, 100)
    edge_complexity = np.sum(edges > 0) / edges.size
    
    # 3. DEFORMAÇÕES - plástico distorce mais a imagem
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    detail_loss = np.mean(np.abs(gray.astype(float) - blur.astype(float)))
    
    return texture_roughness, edge_complexity, detail_loss

def analyze_color_properties(img):
    """Analisa propriedades de cor - importante para vidro colorido"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. SATURAÇÃO - vidro colorido tem saturação uniforme e intensa
    saturation = hsv[:, :, 1]
    sat_mean = np.mean(saturation)
    sat_std = np.std(saturation)
    
    # 2. UNIFORMIDADE DE COR - vidro tem cor mais homogênea
    color_uniformity = 1.0 / (sat_std + 1)  # quanto menor variação, mais uniforme
    
    # 3. INTENSIDADE - vidro costuma ter cores mais puras
    value = hsv[:, :, 2]
    val_mean = np.mean(value)
    
    return sat_mean, sat_std, color_uniformity, val_mean

def analyze_transparency_and_translucency(img):
    """Analisa características de transparência e translucidez"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 1. Distorção ótica
    # Plástico causa mais distorção que vidro
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    distortion_score = np.std(laplacian)
    
    # 2. Padrão de reflexão
    # Vidro tem reflexos mais nítidos e definidos
    _, highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    highlight_sharpness = cv2.Laplacian(highlights, cv2.CV_64F).var()
    
    # 3. Consistência da superfície
    # Vidro tem superfície mais uniforme
    surface_uniformity = cv2.Sobel(gray, cv2.CV_64F, 1, 1).var()
    
    return distortion_score, highlight_sharpness, surface_uniformity

def analyze_surface_smoothness(img):
    """Analisa a suavidade da superfície"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Micro-texturas
    # Plástico tem mais micro-texturas
    kernel = np.ones((3,3), np.float32)/9
    filtered = cv2.filter2D(gray, -1, kernel)
    texture_detail = np.mean(np.abs(gray - filtered))
    
    # 2. Regularidade de bordas
    edges = cv2.Canny(gray, 50, 150)
    edge_smoothness = cv2.blur(edges.astype(float), (5,5)).var()
    
    return texture_detail, edge_smoothness

def predict_crop(model, crop_img):
    """Classifica a imagem e retorna o rótulo e a confiança"""
    # Pré-processamento da imagem
    img_processed = tf.image.resize(crop_img, IMG_SIZE)
    img_processed = img_processed / 255.0
    
    # Obter predições do modelo
    predictions = model.predict(np.expand_dims(img_processed, 0))[0]
    
    # Análises específicas
    distortion, highlights, uniformity = analyze_transparency_and_translucency(crop_img)
    texture, edge_smooth = analyze_surface_smoothness(crop_img)
    
    # Ajuste da predição baseado em características físicas
    if predictions[CLASS_NAMES.index('glass')] > 0.3:
        if (distortion > 0.8 or texture > 0.5):
            predictions[CLASS_NAMES.index('plastic')] += 0.3
            predictions[CLASS_NAMES.index('glass')] -= 0.3
    
    # Normalizar as probabilidades após os ajustes
    predictions = predictions / np.sum(predictions)
    
    # Retornar o rótulo e a confiança
    label = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    
    return label, confidence