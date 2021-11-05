from keras.models import *
from keras.optimizers import Adam
from keras.layers import *
from keras.engine import Model
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception

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
        self.learning_rate_value = learning_rate

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

    def mobilenet(self):

        """

        Function to return MobileNet model

        """

        base_model = MobileNet(weights=self.weights, include_top=self.include_top,
                                 pooling='avg')
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
            epsilon: float = 1e-8,
    ):
        """

        Function to compile the model

        :param learning_rate: Initial learning rate for ADAM optimizer
        :param beta1: Exponential decay rate for the running average of the gradient
        :param beta2: Exponential decay rate for the running average of the square of the gradient
        :param epsilon: Epsilon parameter to prevent division by zero error
        :return: Compiled Keras model

        """

        adam = Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
        self.model_out.compile(
            loss="categorical_crossentropy",
            optimizer=adam,
            metrics=["categorical_accuracy"],
        )

        return self.model_out
