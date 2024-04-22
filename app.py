import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import  ResNet50, preprocess_input

# model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model = ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
