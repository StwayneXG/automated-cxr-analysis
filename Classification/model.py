from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
import tensorflow as tf

from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet169


# def conv_block(input, num_filters):
#     x = Conv2D(num_filters, 3, padding="same", activation="relu")(input)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
#     x = MaxPool2D()(x)

#     return x

""" Initial Model """
# def build_classifier(input_shape, num_classes):
#     inputs = Input(input_shape)

#     c1 = conv_block(inputs, 32)
    
#     c2 = conv_block(c1, 32)
    
#     c3 = conv_block(c2, 64)
    
#     # d1 = Dropout(0.4)(c3)

#     f1 = Flatten()(c3)

#     dense1 = Dense(128, activation="relu")(f1)
#     outputs = Dense(num_classes, activation="softmax")(dense1)

#     model = Model(inputs, outputs, name="Classifier")
#     return model

""" PAPER MODEL """
""" Pre-processing methods in chest X-ray image classification """
# def build_classifier(input_shape, num_classes):
#     inputs = Input(input_shape)

#     conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
#     conv1 = BatchNormalization()(conv1)
#     pool1 = MaxPool2D(pool_size=(2, 2))(conv1)


#     conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
#     conv2 = BatchNormalization()(conv2)
#     pool2 = MaxPool2D(pool_size=(2, 2))(conv2)


#     conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
#     conv3 = BatchNormalization()(conv3)
#     pool3 = MaxPool2D(pool_size=(2, 2))(conv3)


#     conv4 = Conv2D(128, 3, activation='relu', padding='same')(pool3)
#     conv4 = BatchNormalization()(conv4)
#     pool4 = MaxPool2D(pool_size=(2, 2))(conv4)

#     x = Flatten()(pool4)
#     x = Dropout(0.0)(x)

#     x = Dense(512, activation='relu', name='Dense_1', dtype='float32')(x)
#     # x = Dense(64, activation='relu', name='Dense_2', dtype='float32')(x)
#     # x = Dense(8, activation='relu', name='Dense_3', dtype='float32')(x)
#     x = Dense(num_classes, activation='softmax', name='Output', dtype='float32')(x)


#     my_model = Model(inputs=[inputs], outputs=[x])
#     return my_model

""" TRANSFER LEARNING """
""" ResNet50 """
# def build_classifier(input_shape, num_classes):
#     vgg = ResNet50(
#       include_top=False,
#       weights="imagenet",
#       input_shape=input_shape)

#     for layer in vgg.layers:
#       layer.trainable = False

#     x = Flatten()(vgg.output)
#     output = Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs=vgg.input, outputs=output)
#     return model

""" TRANSFER LEARNING """
""" Stacked Models MobileNetV2 + DenseNet169 """
def build_classifier(input_shape, num_classes):
    inputs = Input(input_shape)
    mobilenet_base = MobileNetV2(weights = 'imagenet',input_shape = input_shape,include_top = False)
    densenet_base = DenseNet169(weights = 'imagenet', input_shape = input_shape,include_top = False)
    
    for layer in mobilenet_base.layers:
        layer.trainable =  False
    for layer in densenet_base.layers:
        layer.trainable = False
        
    model_mobilenet = mobilenet_base(inputs)
    model_mobilenet = GlobalAveragePooling2D()(model_mobilenet)
    output_mobilenet = Flatten()(model_mobilenet)

    model_densenet = densenet_base(inputs)
    model_densenet = GlobalAveragePooling2D()(model_densenet)
    output_densenet = Flatten()(model_densenet)

    merged = tf.keras.layers.Concatenate()([output_mobilenet, output_densenet])

    x = BatchNormalization()(merged)
    x = Dense(256,activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128,activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation = 'softmax')(x)
    stacked_model = Model(inputs = inputs, outputs = x)

    return stacked_model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_classifier(input_shape, 6)

    model.summary()