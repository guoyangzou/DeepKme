# DeepKme
DeepKme is the predictor for lysine methylation sites in human proteome. Here is the definition of DeepKme using python.

    import tensorflow as tf
    import pandas as pd
    import numpy as np

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
    model.summary()

    model.load_weights("./Model_split/Km1_CSTCS_3746.hdf5")  

You can use it to make prediction:

    # load the positive samples
    df_Kme = pd.read_csv("./datasets/KmeSites_Collected.csv")  

    # encoding
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

    # make prediction in the positive samples
    y_pred_pos = model.predict(x_data_pos,1500)

    # load the negative samples
    Neg_test = pd.read_csv("./datasets/Negative_samples_for_test.csv")["Negative_samples_for_test"]
    np_data_neg = fun_ser_to_numpy_onehot(Neg_test,0)
    x_data_neg, y_data_neg = np_data_neg[:,:-1],np_data_neg[:,-1]

    # make prediction in the negative samples
    y_pred_neg = model.predict(x_data_neg,1500)

    # evaluate the performance
    y_pred = np.concatenate([y_pred_pos,y_pred_neg])
    y_true = np.concatenate([y_data_pos,y_data_neg])
    result = tf.metrics.AUC(1000)(y_true,y_pred[:,3])
    print("AUC：%.3f"%result)


if you want to replicate the Paper, please install conda first (https://conda.io/en/latest/miniconda.html#windows-installers) and ensure it is successful. 
Then run the following to create an environment and open your jupyter notebook in your browser:

    conda create -n ML2  -c conda-forge -c pytorch python=3.9 pytorch=1.9 tensorflow=2.6 cudnn=8 cudatoolkit=11 scipy pandas openpyxl xlrd jupyterlab jupyter_contrib_nbextensions
    conda activate ML2
    pip install  tensorflow-gpu
    cd [your project dir]
    jupyter notebook

Open the DeepKme.ipynb in your browser where your jupyter notebook works. 
Then your can run each cell to replicate the paper's work.
