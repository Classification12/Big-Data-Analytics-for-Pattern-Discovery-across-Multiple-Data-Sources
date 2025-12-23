import numpy as np
import pandas as pd
from Model_CPR import Model_CPR
from Model_FLSA import Model_FLSA
from Model_Kmeans import Model_Kmeans
from Model_RBilSTM import Model_RBiLSTM
from Model_SARBiLSTM_NLAF import Model_SARBiLSTM_NLAF
from PlotResults import *

# Read Dataset 1
an = 0
if an == 1:
    Filename = np.asarray(pd.read_csv('./Dataset/CensusData.csv'))
    Data = Filename[:, :-1]
    Target = Filename[:, -1]
    np.save('Data_1.npy', Data)
    np.save('Target_1.npy', Data)

# Read Dataset 2
an = 0
if an == 1:
    Filename = np.asarray(pd.read_csv('./Dataset/kddcup99_csv.csv'))
    Data = Filename[:, :-1]
    Target = Filename[:, -1]
    np.save('Data_2.npy', Data)
    np.save('Target_2.npy', Data)

# Read Dataset 3
an = 0
if an == 1:
    Filename = np.asarray(pd.read_csv('./Dataset/higgs.csv'))
    Data = Filename[:, :-1]
    Target = Filename[:, -1]
    np.save('Data_3.npy', Data)
    np.save('Target_3.npy', Data)

# Read Dataset 4
an = 0
if an == 1:
    Filename = np.asarray(pd.read_csv('./Dataset/susy.csv'))
    Data = Filename[:, :-1]
    Target = Filename[:, -1]
    np.save('Data_4.npy', Data)
    np.save('Target_4.npy', Data)

No_of_Dataset = 4

# Pattern Formation
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Feature = np.load('Data_' + str(n+1) + '.npy', allow_pickle=True)  # loading step
        Target = np.load('Target_' + str(n+1) + '.npy', allow_pickle=True)  # loading step
        Feat = Feature
        EVAL = []
        Batch_Size = [16, 32, 64, 128, 256]
        for act in range(len(Batch_Size)):
            learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((5, 11))
            Eval[0, :], pred = Model_Kmeans(Train_Data, Train_Target, Test_Data, Test_Target,
                                            Batch_Size[act])  # Model Kmeans
            Eval[1, :], pred1 = Model_FLSA(Train_Data, Train_Target, Test_Data,
                                                Test_Target, Batch_Size[act])  # Model FLSA
            Eval[2, :], pred2 = Model_CPR(Train_Data, Train_Target, Test_Data, Test_Target,
                                                Batch_Size[act])  # Model CPR
            Eval[3, :], pred3 = Model_RBiLSTM(Train_Data, Train_Target, Test_Data,
                                             Test_Target, Batch_Size[act])  # RBi-LSTM
            Eval[4, :], pred4 = Model_SARBiLSTM_NLAF(Train_Data, Train_Target, Test_Data,
                                             Test_Target, Batch_Size[act])  # Model_SARBiLSTM_NLAF
            EVAL.append(Eval)
        np.save('Evaluate_all.npy', EVAL)  # Save Eval all

Plot_Results_Dataset()
Plot_Results()
Plot_ROC_Curve()
Table()