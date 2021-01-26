from tensorflow.keras.applications import resnet_v2

def classification_model(mattr, **kwargs):
    model = resnet_v2.ResNet50V2(include_top=True, weights=None,
    input_shape=(mattr.inheight, mattr.inwidth, 3), classes=mattr.nclasses)
    return model