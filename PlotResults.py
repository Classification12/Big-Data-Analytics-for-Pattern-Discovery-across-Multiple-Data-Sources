import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.patheffects as pe

No_of_Dataset = 4


def Plot_ROC_Curve():
    cls = ['KMC', 'F-LSA', 'CPR', 'RBiLSTM', 'SARBiLSTM-NLAF']
    for a in range(No_of_Dataset):  # For 2 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.75)
        # Actual = Actual[lenper:, :]
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset - ' + str(a + 1) + ' - ROC Curve')
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
        for i, color in zip(range(len(cls)), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i].astype('float64')
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)',
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy', fontsize=14)
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("ROC Curve")
        plt.legend(loc="lower right", fontsize=14)
        path = "./Results/ROC_%s.png" % (a + 1)
        plt.savefig(path)
        plt.show()


def Table():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Classifier = ['TERMS', 'KMC', 'F-LSA', 'CPR', 'RBiLSTM', 'SARBiLSTM-NLAF']
    Terms = ['Accuracy', 'Recall', 'Precision', 'Confidence', 'Growth Rate', 'Purity Ratio', 'NMI Ratio']
    Table_Terms = [0, 1, 2, 3, 4, 5, 6]
    table_terms = [Terms[i] for i in Table_Terms]
    Epoch = ['50', '100', '150', '200']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Classifier[0], Epoch)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j, k])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Method Comparison',
                  '---------------------------------------')
            print(Table)


def Plot_Results_Dataset():
    eval = np.load('Evaluate_all_Dataset.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Recall', 'Precision', 'Confidence', 'Growth Rate', 'Purity Ratio', 'NMI Ratio']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6]
    Dataset = [1, 2, 3, 4]
    # for i in range(eval.shape[0]):
    for j in range(len(Graph_Terms)):
        Graph = np.zeros((eval.shape[0], eval.shape[2]))
        # for k in range(eval.shape[0]):
        for l in range(eval.shape[2]):
            Graph[:, l] = eval[:, 4, l, Graph_Terms[j] + 4]

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        # ax.yaxis.grid()
        # ax.set_facecolor("#e0f3db")
        # fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Algorithm Comparison of BatchSize')
        plt.plot(Dataset, Graph[:, 0], lw=5, color='blue',
                 path_effects=[pe.withStroke(linewidth=13, foreground='violet')], marker='h',
                 markerfacecolor='blue', markersize=5,
                 label="KMC")
        plt.plot(Dataset, Graph[:, 1], lw=5, color='maroon',
                 path_effects=[pe.withStroke(linewidth=13, foreground='tan')], marker='h',
                 markerfacecolor='#7FFF00', markersize=5,
                 label="F-LSA")
        plt.plot(Dataset, Graph[:, 2], lw=5, color='lime',
                 path_effects=[pe.withStroke(linewidth=13, foreground='orange')], marker='h',
                 markerfacecolor='#808000',
                 markersize=5,
                 label="CPR")
        plt.plot(Dataset, Graph[:, 3], lw=10, color='deeppink',
                 path_effects=[pe.withStroke(linewidth=13, foreground='w')], marker='h', markerfacecolor='#CD1076',
                 markersize=9,
                 label="RBiLSTM")
        plt.plot(Dataset, Graph[:, 4], lw=5, color='k',
                 path_effects=[pe.withStroke(linewidth=13, foreground='red')], marker='h', markerfacecolor='black',
                 markersize=5,
                 label="SARBiLSTM-NLAF")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
                   ncol=3, fancybox=True)
        plt.xticks(Dataset, ('census', 'kddcup', 'higgs', 'susy'), fontsize=14)
        plt.xlabel('Datasets', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
                   ncol=14, fancybox=True)
        path = "./Results/Dataset Comparison - %s.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()


def Plot_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Recall', 'Precision', 'Confidence', 'Growth Rate', 'Purity Ratio', 'NMI Ratio']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            import pandas as pd
            act = [0, 1, 2, 3, 4]
            Graph1 = Graph[:, :5]
            df1 = pd.DataFrame(Graph1)
            ax = df1.plot(kind='bar', legend=False, figsize=(10, 6), rot=0, width=0.8)

            bars = ax.patches
            hatches = ''.join(h * len(df1) for h in 'x/O.*')

            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
            plt.legend(['KMC', 'F-LSA', 'CPR', 'RBiLSTM', 'SARBiLSTM-NLAF'], loc='upper center', bbox_to_anchor=(0.5, 1.10),
                       ncol=5, fontsize=14, fancybox=True)
            plt.xticks(act, ('16', '32', '64', '128', '256'), fontname="Arial", fontsize=14)
            plt.xlabel('Batch Size', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            # plt.ylim([65, 95])
            path = "./Results/Activation Function Variation - Dataset-%s-%s.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    # Plot_Results_Dataset()
    Plot_Results()
    Plot_ROC_Curve()
    Table()
