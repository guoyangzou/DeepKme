All the trained models' weights saved in file type hdf5. Before load the file, please run in python:
##############################################################################
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.compile(optimizer=tf.optimizers.Adam()
                     ,loss=tf.losses.BinaryCrossentropy()
                     ,metrics=[tf.metrics.AUC(1000)]
                    )

        self.cnn1 = tf.keras.Sequential([
            tf.keras.layers.Reshape([61,21]),
            tf.keras.layers.Conv1D(256,9,1,"valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Dropout(0.7),
        ])
        self.cnn2 = tf.keras.Sequential([
            tf.keras.layers.Reshape([26,256]),
            tf.keras.layers.Conv1D(32,7,1,"valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Dropout(0.5),
        ])
        self.simple = tf.keras.Sequential([
            tf.keras.layers.Reshape([10*32]),
            tf.keras.layers.Dense(128),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(4,activation="sigmoid"),
        ])
    def call(self, inputs):

        x = self.cnn1(inputs)
        x = self.cnn2(x)
        x = self.simple(x)

        return x
model = Model()
model.build((None,61*21))

################################################################################

