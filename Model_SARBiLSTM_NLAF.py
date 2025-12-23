import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Add, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from Evaluation import evaluation


# ---- 1. Spatial Attention Layer ----
class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1],), initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[1],), initializer="zeros", trainable=True)
        super(SpatialAttention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, K.expand_dims(self.W)) + self.b)
        a = K.softmax(e)
        return K.sum(x * K.expand_dims(a), axis=1)  # Weighted sum


# ---- 2. Novel Activation Function (NLAF) ----
def novel_activation(x):
    return K.maximum(K.sigmoid(x), K.tanh(x))  # max(Sigmoid(x), Tanh(x))


# ---- 3. Novel Loss Function (Modified Huber Loss) ----
def modified_huber_loss(y_true, y_pred):
    xy = y_true * y_pred
    loss = tf.where(xy > -1, K.square(K.maximum(0.0, 1 + xy)), -8 * xy)
    return K.mean(loss)


def Model_SARBiLSTM_NLAF(Train_Data, Train_Target, Test_Data, Test_Target, batch):
    # ---- 5. Example Usage ----
    input_shape = Train_Data.shape
    inputs = Input(shape=input_shape)
    units = 128
    dropout_rate = 0.3

    # First BiLSTM Layer
    x = Bidirectional(LSTM(units, return_sequences=True))(inputs)

    # Second BiLSTM with Residual Connection
    x_residual = Bidirectional(LSTM(units, return_sequences=True))(x)
    x = Add()([x, x_residual])  # Residual connection

    # Spatial Attention Layer
    x = SpatialAttention()(x)

    # Fully Connected Layer with Novel Activation
    x = Dense(64, activation=novel_activation)(x)

    # Output Layer (for classification, use sigmoid; for regression, use linear)
    outputs = Dense(1, activation="sigmoid")(x)

    # Compile the Model
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss=modified_huber_loss, metrics=["accuracy"])
    model.fit(Train_Data, Train_Target, epochs=5, steps_per_epochs=10, batch_size=batch)
    pred = model.predict(Test_Data)
    Eval = evaluation(pred, Test_Target)
    return pred, Eval
