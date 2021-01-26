import tensorflow
import tensorflow.keras.models as kmodels
from . import layers
import tensorflow.keras.layers as klayers
from tensorflow.keras.initializers import glorot_normal

import metrics_and_losses
tracker = []


def custom_encoder(img_input, mattr, **kwargs):
    channel_multiplier = kwargs['channel_multiplier']
    channels = int(32 * channel_multiplier)
    # padding = "same"
    if('padding') in kwargs.keys():
        padding = kwargs['padding']

    x = img_input
    connections = []
    encoder_name = kwargs['encoder']
    dropout_rate = kwargs['dropout_rate']

    encoder_fn = getattr(layers, encoder_name)

    layer_num = 0

    # add encoder layers until this image height is reached
    # minimum_height = 6
    # while(x.shape[1] >= minimum_height * 2):
    double_convs_until = kwargs['double_convs_until']

    for _ in range(kwargs['down_blocks']):
        # pdb.set_trace()
        if kwargs['decoder'] == "up_unet":
            if kwargs['down_blocks'] - layer_num <= kwargs['up_blocks']:
                padding = 'same'

        # determine which blocks consist
        # of two instead of one convolutional block
        if(double_convs_until >= 0):
            double_conv = layer_num >= abs(double_convs_until)
        else:
            double_conv = kwargs['down_blocks'] -\
                layer_num <= abs(double_convs_until)
        x, connection = encoder_fn(layer_num, x, channels, padding,
                                   double_conv=double_conv,
                                   activation=kwargs['activation'])

        if(connection is not None):
            connections.append(connection)
        channels *= 2
        layer_num += 1

    channels = int(channels / 2)
    x = klayers.SeparableConv2D(
        channels, (3, 3),
        activation=kwargs['activation'],
        padding="same",
        name='down_block%i_conv1_' % layer_num,
        kernel_initializer=glorot_normal())(x)
    x = klayers.Dropout(dropout_rate, name='middle_dropout')(x)

    if(len(connections) == 0):
        connections = None

    return x, channels, connections


def append_decoder(x, channels, mattr, connections=None, **kwargs):
    """
    takes a functional model of the encoder and appends the decoder
    """

    decoder_blocks = kwargs['up_blocks']
    # encoder_blocks = kwargs['down_blocks']
    decoder_name = kwargs['decoder']
    padding = kwargs['padding']

    decoder_fn = getattr(layers, decoder_name)

    for layer_num in range(decoder_blocks):
        if(connections):
            connection = connections[len(connections) - layer_num - 1]
        else:
            connection = None
        print("connections: ", connections)
        x = decoder_fn(layer_num, x, channels, padding, connection)
        channels = channels//4

    x = klayers.SeparableConv2D(mattr.nclasses, (3, 3), padding=padding,
                                data_format='channels_last')(x)
    return x


def keras_applications_encoder(img_input, mattr, **kwargs):
    encoder_fn = getattr(tensorflow.keras.applications,
                         kwargs['encoder'])
    options = {}
    if "MobileNet" in kwargs['encoder']:
        options['alpha'] = kwargs['keras_alpha']
        options['weights'] = kwargs["keras_weights"]
        if options['weights'] == "None":
            options['weights'] = None
    minimum_height = 4
    encoder_model = encoder_fn(
        input_shape=[mattr.inheight, mattr.inwidth, 3],
        include_top=False, **options)
    # encoder_model.summary()
    tensorflow.keras.utils.plot_model(
        encoder_model, to_file="model.png",
        show_shapes=True, show_layer_names=True)
    layers = []

    for layer in encoder_model.layers:
        if "EfficientNet" in kwargs['encoder']:
            is_last = "expand_activation" in layer.name
            if is_last and not layer.output.shape[1] >= minimum_height * 2:
                break
        else:
            if not layer.output.shape[1] >= minimum_height * 2:
                break
        layers.append(layer.output)

    # Create the feature extraction model
    down_stack = tensorflow.keras.Model(inputs=encoder_model.input,
                                        outputs=layers)
    down_stack.trainable = True

    x = down_stack(img_input)[-1]
    connections = None
    channels = 512
    return x, channels, connections


def SegmentationModel(mattr, **kwargs):
    img_input = klayers.Input(shape=(mattr.inheight, mattr.inwidth, 3))

    if kwargs['encoder'] not in ["down_unet", "down_plain"]:
        x, channels, connections = keras_applications_encoder(
            img_input, mattr, **(kwargs))
    else:
        x, channels, connections = custom_encoder(img_input, mattr, **(kwargs))

    channels = int(channels * kwargs['channel_decoder_multiplier'])

    x = append_decoder(x, channels, mattr, connections, **(kwargs))

    o_shape = kmodels.Model(img_input, x).output_shape
    outheight, outwidth = o_shape[1], o_shape[2]
    x = klayers.Reshape((outheight*outwidth, mattr.nclasses))(x)

    if(not mattr.from_logits):
        x = (klayers.Activation('softmax', name='softmax'))(x)
    model = kmodels.Model(inputs=img_input, outputs=x)
    model.outheight = outheight
    model.outwidth = outwidth

    return model
