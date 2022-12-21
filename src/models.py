from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.xception import Xception


class Models(object):
    """

    Class to compile different pre-trained CNN models

    """

    def __init__(
            self,
            trainable,
            num_classes,
            weights,
            include_top,
            learning_rate
    ):
        """

        :param trainable: Number of layers to be trained
        :param num_classes: Number of classes in the dataset
        :param weights: Pre-trained wights of the models
        :param include_top: Whether or not to include the top layer of the pre-trained model

        """

        self.trainable = trainable
        self.num_classes = num_classes
        self.weights = weights
        self.include_top = include_top
        self.learning_rate = learning_rate

        self.model_out = None

    def densenet121(self):
        """

        Function to return DenseNet121 model

        """

        base_model = DenseNet121(weights=self.weights, include_top=self.include_top,
                                 pooling='avg')
        x = base_model.layers[-1]
        out = Dense(units=self.num_classes, activation='softmax', name='output',
                    use_bias=True)(x.output)

        self.model_out = Model(inputs=base_model.input, outputs=out)

    def mobilenet(self, input_shape):
        """

        Function to return MobileNet model

        """

        base_model = MobileNet(weights=self.weights, include_top=self.include_top,
                               pooling='avg', input_shape=input_shape)
        x = base_model.layers[-1]
        out = Dense(units=self.num_classes, activation='softmax', name='output',
                    use_bias=True)(x.output)

        self.model_out = Model(inputs=base_model.input, outputs=out)

    def xception(self):
        """

        Function to return Xception model

        """

        base_model = Xception(weights=self.weights, include_top=self.include_top,
                              pooling='avg')
        x = base_model.layers[-1]
        out = Dense(units=self.num_classes, activation='softmax', name='output',
                    use_bias=True)(x.output)

        self.model_out = Model(inputs=base_model.input, outputs=out)

    def model_trainable(self):
        """

        Function to define if the layers should be trainable or not

        """

        for layer in self.model_out.layers:
            layer.trainable = self.trainable

    def model_compile(
            self,
            learning_rate,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-20,
            decay: float = 0.0000001,
            amsgrad = False
    ):
        """

        Function to compile the model

        :param learning_rate: Initial learning rate for ADAM optimizer
        :param beta1: Exponential decay rate for the running average of the gradient
        :param beta2: Exponential decay rate for the running average of the square of the gradient
        :param epsilon: Epsilon parameter to prevent division by zero error
        :return: Compiled Keras model

        """

        adam = Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
        self.model_out.compile(
            loss="categorical_crossentropy",
            optimizer=adam,
            metrics=["categorical_accuracy"],
        )

        return self.model_out
