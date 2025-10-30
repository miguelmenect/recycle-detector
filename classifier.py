# treina e faz uso do modelo para cliassifcar em uma das categorias
#tensorflow lib de treino, serve para criar e treinar redes neurais(pensa e aprende como cerebro)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
#definindo tamanho padrão das imagens
IMG_SIZE = (224, 224)
BATCH_SIZE = 4  # menor batch para dataset pequeno, mostra no escopo de duas fotos por 
                #vez para o modelo de treino treinar
#clasificações
CLASS_NAMES = ["glass", "metal", "organic", "paper", "plastic", "others"]

def build_model(num_classes=len(CLASS_NAMES)): #cerebro artificial que vai aprender a classificar os objetos
                                               # em categorias de reciclaveis
    base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                             include_top=False,
                                             weights='imagenet')
    base.trainable = False ##para não alterar o que o modelo já sabe, mantendo seu repertório atual
                           # de conhecimento sobre formas e cores e etc durante treino
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)

    #produz a % de chance de que a imagem pode ter de cada 
    #categoria, (exemplo: plastico: "10%", papel "65%", metal: "15%" e etc )
    out = layers.Dense(num_classes, activation='softmax')(x) #saida/resposta do treino
    
    model = models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#função que treina modelo com as imagens de treino
def train(data_dir, model_out="models/waste_classifier.h5", epochs=30):
    # ajustes de proporção da imagem para RECONHECER melhor objetos em 
    # imagens e RECONHECE-LOS individualmente
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,          
        width_shift_range=0.3,       
        height_shift_range=0.3,      
        zoom_range=0.2,              
        brightness_range=[0.8, 1.2],
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

def predict_crop(model, crop_img):
    # crop_img: BGR numpy (OpenCV). Resize + normalize + predict
    img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    x = img.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = preds.argmax()
    return CLASS_NAMES[idx], float(preds[idx])
