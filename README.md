# DeepKme
DeepKme is the predictor for lysine methylation sites in human proteome. Here is the definition of DeepKme using python.

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
When using the the model to make prediction, we can use:
    
    import pandas as pd
    model.load_weights("C:/Users/zou/Downloads/0_0.hdf5")  # "C:/Users/zou/Downloads/0_0.hdf5" is your model's parameter file.
    df_Kme = pd.read_csv("C:/Users/zou/Downloads/KmeSites_Collected.txt")  # please replace it with your file.
    
    def fun_ser_to_numpy_onehot(Se,label):
        AAs = ['Q', 'L', 'N', 'G', 'R', 'F', '_', 'W', 'T', 'E', 'K', 'I', 'D', 'V', 'Y', 'S', 'A', 'C', 'M', 'H', 'P']
        Se = Se.copy()
        df = Se.drop_duplicates().apply(lambda x: pd.Series(list(x))).replace(AAs,range(len(AAs))).copy()
        df["label"] = label
        se = df.to_numpy()
        se_x = tf.one_hot(se[:,:-1],21).numpy().reshape([len(se),61*21])
        se_y = se[:,-1:]

        return np.concatenate([se_x,se_y],1)
        
    df_Kme_Sites = df_Kme.SeqWin
    np_data_pos = fun_ser_to_numpy_onehot(df_Kme_Sites,1)
    x_data_pos, y_data_pos = np_data_pos[:,:-1],np_data_pos[:,-1]
    y_pred_pos = model.predict(x_data_pos,1500)

    Neg_test = pd.read_csv("Negative_samples.txt")["0"]
    np_data_neg = fun_ser_to_numpy_onehot(Neg_test,0)
    x_data_neg, y_data_neg = np_data_neg[:,:-1],np_data_neg[:,-1]
    y_pred_neg = model.predict(x_data_neg,1500)
    
    y_pred = np.concatenate([y_pred_pos,y_pred_neg])
    y_true = np.concatenate([y_data_pos,y_data_neg])
    tf.metrics.AUC(1000)(y_true,y_pred[:,3])
