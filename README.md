# CNNArginineMe

CNNArginineMe is the predictor for lysine methylation sites in human proteome.

This package contains five folders and the jupyter notebook file CNNArginineMe.ipynb, described as follows.

CNNArginineMe.ipynb contain the codes for the data preprocessing, model construction and the experiment-split test.

The ID_convert_list folder inlcudes the original datasets derived from PhosphoSite, each of which contains the information of the Kme types and the experiment source. All the data are integrated as PhosphoSitePlus_withEvidence.csv, stored under the datasets folder.

the Model_split folder contains the CNNArginineMe models with the hdf5 format. The models were constructed using experiment-split method (See Figure 2).

The data folder contains temporary files generated during the running of the jupyter notebook for the experiment-split method. These files can be ignored by users.

The datasets folder contains the positive and negative datasets as well as the data files used to build both datasets. Please see the readme.txt under this folder for details. 

The orig_dataset contains the Kme information directly downloaded from different databases. These files were later proprocessed and the final files were stored in the datasets folder for the generation of positive and negative samples. 

The following is the definition of CNNArginineMe using python.

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

You can use it to make prediction:
    
    # Load weights to the defined model
    model.load_weights("./Model_split/Km1_CSTCS_3746.hdf5")  ## you can load any model weights as you need in Model_split folder.
    
    # Load the positive samples
    df_Kme = pd.read_csv("./datasets/KmeSites_Collected.csv")  ## you can prepare your own unlabled datasets for predicting the Km1/2/3/e score.

    # Encoding
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
    x_data_pos = np_data_pos[:,:-1]
    y_data_pos = np_data_pos[:,-1]  ## If you just make prediction for your own unlabled datasets, this step should be passed.

    # Make prediction in the positive samples
    y_pred_pos = model.predict(x_data_pos)  ## If you make prediction for your own unlabled datasets, this step would get the Km1/2/3/e score.

    # Load the negative samples
    Neg_test = pd.read_csv("./datasets/Negative_samples_for_test.csv")["Negative_samples_for_test"]
    np_data_neg = fun_ser_to_numpy_onehot(Neg_test,0)
    x_data_neg, y_data_neg = np_data_neg[:,:-1],np_data_neg[:,-1]

    # Make prediction in the negative samples
    y_pred_neg = model.predict(x_data_neg)

    # Evaluate the performance. If you just make prediction for your own unlabled datasets, this step should be passed.
    y_pred = np.concatenate([y_pred_pos,y_pred_neg])
    y_true = np.concatenate([y_data_pos,y_data_neg])
    result = tf.metrics.AUC(1000)(y_true,y_pred[:,3])
    print("AUC：%.3f"%result)


If you want to replicate the paper, please install conda first (https://conda.io/en/latest/miniconda.html#windows-installers) and ensure it is successful. 
Then run the following to create an environment and open your jupyter notebook in your browser:

    # terminal
    conda create -n ML2  -c conda-forge -c pytorch python=3.9 pytorch=1.9 tensorflow=2.6 cudnn=8 cudatoolkit=11 scipy pandas openpyxl xlrd jupyterlab jupyter_contrib_nbextensions
    conda activate ML2
    pip install  tensorflow-gpu==2.6  ## if your computer have no gpu, ignore this step.
    cd [your project dir]
    jupyter notebook

Open the DeepKme.ipynb in your browser where your jupyter notebook works. 
