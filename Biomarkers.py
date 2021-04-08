import pathlib
import itertools
import warnings
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from utils import convertMatlabToDataframe
from utils import add_labels_to_df
from utils import get_classification
from utils import find_best_features

warnings.filterwarnings("ignore")


class Biomarkers:
    def __init__(self):
        self.rowIndices = [18, 49, 10, 41, 2, 33, 8, 39, 4, 35, 26, 57, 14, 45, 19, 50, 21, 52, 53, 22, 7, 38, 5, 36,
                           23, 54,
                           13, 44, 11, 42, 3, 34, 20, 51, 30, 61, 29, 60, 0, 31, 46, 59, 27, 58, 12, 43, 6, 37, 1, 25,
                           32,
                           56, 9, 40, 16, 47, 48, 15, 17, 28, 24, 55]
        self.colIndices = [18, 49, 10, 41, 2, 33, 8, 39, 4, 35, 26, 57, 14, 45, 19, 50, 21, 52, 53, 22, 7, 38, 5, 36,
                           23, 54,
                           13, 44, 11, 42, 3, 34, 20, 51, 30, 61, 29, 60, 0, 31, 46, 59, 27, 58, 12, 43, 6, 37, 1, 25,
                           32,
                           56, 9, 40, 16, 47, 48, 15, 17, 28, 24, 55]

        self.native_list = []
        self.vulnerableList = []
        self.resilientList = []
        self.controlList = []

        self.cluster_1 = []
        self.cluster_2 = []
        self.cluster_3 = []
        self.cluster_4 = []
        self.cluster_5 = []

        self.cluster_region_list = []

        self.sumEl = 0
        self.mean = 0

        self.numpy_native_3d = np.zeros(shape=(62, 62))
        self.vul_numpy_native_3d = np.zeros(shape=(62, 62))
        self.res_numpy_native_3d = np.zeros(shape=(62, 62))
        self.control_numpy_native_3d = np.zeros(shape=(62, 62))

        self.numpy_mni_3d = np.zeros(shape=(62, 62))
        self.vul_numpy_mni_3d = np.zeros(shape=(62, 62))
        self.res_numpy_mni_3d = np.zeros(shape=(62, 62))
        self.control_numpy_mni_3d = np.zeros(shape=(62, 62))

        self.subjects = ['Subject001', 'Subject002', 'Subject003', 'Subject004', 'Subject006', 'Subject007',
                         'Subject008', 'Subject009', 'Subject010', 'Subject011', 'Subject012', 'Subject013',
                         'Subject014',
                         'Subject015', 'Subject016', 'Subject017', 'Subject018', 'Subject019', 'Subject020',
                         'Subject021',
                         'Subject022', 'Subject023', 'Subject024', 'Subject025', 'Subject026', 'Subject027',
                         'Subject028',
                         'Subject029', 'Subject030', 'Subject031', 'Subject032', 'Subject033', 'Subject034',
                         'Subject035',
                         'Subject036', 'Subject037', 'Subject038', 'Subject039', 'Subject040', 'Subject041',
                         'Subject042',
                         'Subject043', 'Subject044', 'Subject045', 'Subject046']

        # A stands for \\rdcemea-t3.storage.philips.com\011497_mri_eindhoven_ux\Resilience_project_with_LUMC\
        self.rootDirPath = pathlib.Path(r'A:\code')
        self.subjectList = pd.read_csv(r'A:\data\bids\participants_updated.tsv', sep='\t')
        self.labelList = pd.read_csv(r'A:\data\bids\bids_native_final\sub_ses-1_T1w_vwbs_atlas.csv', sep=',')

        for index, row in self.subjectList.iterrows():
            # print(row)
            if row['diagnosis'] == 'vulnerable':
                self.vulnerableList.append(row['subject_id'])
            elif row['diagnosis'] == 'resilient':
                self.resilientList.append(row['subject_id'])
            else:
                self.controlList.append(row['subject_id'])

        for subject in self.subjects:
            self.mat_file_native = loadmat(pathlib.Path(self.rootDirPath / 'conn_project_native' /
                                                        'results' / 'firstlevel' / 'SBC_01'
                                                        / ('resultsROI_' + subject + '_Condition001.mat')))

            self.mat_file_mni = loadmat(pathlib.Path(self.rootDirPath / 'matlab' / 'conn_LUMC_dataset1' /
                                                     'results' / 'firstlevel' / 'SBC_01'
                                                     / ('resultsROI_' + subject + '_Condition001.mat')))

            self.df_file_native = convertMatlabToDataframe(self.mat_file_native)
            self.numpy_native = self.df_file_native.to_numpy()

            self.df_file_mni = convertMatlabToDataframe(self.mat_file_mni)
            self.numpy_mni = self.df_file_mni.to_numpy()

            # Native space
            if (self.numpy_native_3d == 0).all():
                # print(' Native Empty')
                self.numpy_native_3d = self.numpy_native
            else:
                self.numpy_native_3d = np.dstack([self.numpy_native_3d, self.numpy_native])

            if subject in self.vulnerableList:
                if (self.vul_numpy_native_3d == 0).all():
                    # print('Native Vulnerable Empty')
                    self.vul_numpy_native_3d = self.numpy_native
                else:
                    self.vul_numpy_native_3d = np.dstack([self.vul_numpy_native_3d, self.numpy_native])
            elif subject in self.resilientList:
                if (self.res_numpy_native_3d == 0).all():
                    # print('Native Resilient Empty')
                    self.res_numpy_native_3d = self.numpy_native
                else:
                    self.res_numpy_native_3d = np.dstack([self.res_numpy_native_3d, self.numpy_native])
            else:
                if (self.control_numpy_native_3d == 0).all():
                    # print('Native Control Group Empty')
                    self.control_numpy_native_3d = self.numpy_native
                else:
                    self.control_numpy_native_3d = np.dstack([self.control_numpy_native_3d, self.numpy_native])

                # MNI Space
                if (self.numpy_mni_3d == 0).all():
                    # print('MNI Empty')
                    self.numpy_mni_3d = self.numpy_mni
                else:
                    self.numpy_mni_3d = np.dstack([self.numpy_mni_3d, self.numpy_mni])

            if subject in self.vulnerableList:
                if (self.vul_numpy_mni_3d == 0).all():
                    # print(' MNI Vulnerable Empty')
                    self.vul_numpy_mni_3d = self.numpy_mni
                else:
                    self.vul_numpy_mni_3d = np.dstack([self.vul_numpy_mni_3d, self.numpy_mni])
            elif subject in self.resilientList:
                if (self.res_numpy_mni_3d == 0).all():
                    # print(' MNI Resilient Empty')
                    self.res_numpy_mni_3d = self.numpy_mni
                else:
                    self.res_numpy_mni_3d = np.dstack([self.res_numpy_mni_3d, self.numpy_mni])
            else:
                if (self.control_numpy_mni_3d == 0).all():
                    # print('MNI Control Group Empty')
                    self.control_numpy_mni_3d = self.numpy_mni
                else:
                    self.control_numpy_mni_3d = np.dstack([self.control_numpy_mni_3d, self.numpy_mni])

        self.meanMatrix_native = np.mean(self.numpy_native_3d, axis=2)
        self.vul_meanMatrix_native = np.mean(self.vul_numpy_native_3d, axis=2)
        self.res_meanMatrix_native = np.mean(self.res_numpy_native_3d, axis=2)
        self.ctrl_meanMatrix_native = np.mean(self.control_numpy_native_3d, axis=2)

        self.meanMatrix_mni = np.mean(self.numpy_mni_3d, axis=2)
        self.vul_meanMatrix_mni = np.mean(self.vul_numpy_mni_3d, axis=2)
        self.res_meanMatrix_mni = np.mean(self.res_numpy_mni_3d, axis=2)
        self.ctrl_meanMatrix_mni = np.mean(self.control_numpy_mni_3d, axis=2)

        self.meanDF_native = pd.DataFrame(self.meanMatrix_native)
        self.meanDF_native = add_labels_to_df(self.meanDF_native)

        self.meanDF_mni = pd.DataFrame(self.meanMatrix_mni)
        self.meanDF_mni = add_labels_to_df(self.meanDF_mni)

        self.mean_vul_meanMatrix_native = pd.DataFrame(self.vul_meanMatrix_native)
        self.mean_vul_meanMatrix_native = self.mean_vul_meanMatrix_native.reindex(self.rowIndices,
                                                                                  columns=self.colIndices)
        self.mean_vul_meanMatrix_native = add_labels_to_df(self.mean_vul_meanMatrix_native)

        self.mean_res_meanMatrix_native = pd.DataFrame(self.res_meanMatrix_native)
        self.mean_res_meanMatrix_native = self.mean_res_meanMatrix_native.reindex(self.rowIndices,
                                                                                  columns=self.colIndices)
        self.mean_res_meanMatrix_native = add_labels_to_df(self.mean_res_meanMatrix_native)

        self.mean_ctrl_meanMatrix_native = pd.DataFrame(self.ctrl_meanMatrix_native)
        self.mean_ctrl_meanMatrix_native = self.mean_ctrl_meanMatrix_native.reindex(self.rowIndices,
                                                                                    columns=self.colIndices)
        self.mean_ctrl_meanMatrix_native = add_labels_to_df(self.mean_ctrl_meanMatrix_native)

        self.meanDiff_vulresMinus_native = self.mean_vul_meanMatrix_native - self.mean_res_meanMatrix_native

    def viz_group_adjacency_matrices(self):
        plt.rcParams["figure.figsize"] = (20, 5)

        plt.subplot(1, 3, 1)
        sns.heatmap(self.mean_vul_meanMatrix_native, center=0)
        plt.title('Vulnerable Mean Matrix in Native')

        plt.subplot(1, 3, 2)
        sns.heatmap(self.mean_res_meanMatrix_native, center=0)
        plt.title('Resilient Mean Matrix in Native')

        plt.subplot(1, 3, 3)
        sns.heatmap(self.meanDiff_vulresMinus_native, center=0, vmin=-0.3, vmax=0.15, cmap="viridis")
        plt.title('Vulnerable minus Resilient Mean Matrix in Native')

        plt.show()

    def prepare_dataset(self, save_dataset):
        list_range = []
        cntr = 0
        data_add = pd.DataFrame()

        data = pd.read_csv(r"A:\data\bids\participants_groups_for_conn.csv", index_col='id_conn_sub')
        data = data[['participant_id', 'sex', 'age', 'diagnosis']]
        stuff = list(self.labelList['brain_regions'])

        for L in range(0, len(stuff) + 1):
            for subset in itertools.combinations(stuff, L):
                if len(subset) == 2:
                    list_range.append(subset)
                    cntr += 1
                    # print(subset)
                if len(subset) > 2:
                    break
        df1 = pd.DataFrame(zip(list_range))
        df1 = df1.rename(columns={0: "adj_brain_regions"})

        for subject in self.subjects:
            train_mat_file_native = loadmat(pathlib.Path(self.rootDirPath / 'conn_project_native' /
                                                         'results' / 'firstlevel' / 'SBC_01'
                                                         / ('resultsROI_' + subject + '_Condition001.mat')))

            df_file_native = convertMatlabToDataframe(train_mat_file_native)
            numpy_native = df_file_native.to_numpy()
            numpy_native = np.nan_to_num(numpy_native)
            numpy_native = np.triu(numpy_native)
            numpy_native = numpy_native.flatten()
            numpy_native = numpy_native[numpy_native != 0]

            np_series = pd.Series(numpy_native)

            data_add1 = pd.DataFrame([np_series], index=[subject])
            data_add = pd.concat([data_add, data_add1])

        for column in data_add:
            data_add = data_add.rename(columns={column: df1['adj_brain_regions'][column]})

        result = pd.concat([data, data_add], axis=1, sort=False)
        result = result.drop(['Subject005'])
        result['sex'] = result['sex'].replace('female', 0)
        result['sex'] = result['sex'].replace('male', 1)
        result = result[result['diagnosis'] != 'control']
        result['diagnosis'] = result['diagnosis'].replace('vulnerable', 0)
        result['diagnosis'] = result['diagnosis'].replace('resilient', 1)
        # result
        if save_dataset:
            result.to_csv('original_dataset_with_participant_features.csv', encoding='utf-8')

        X = result
        Y = result['diagnosis']
        X = X.drop(['participant_id', 'diagnosis', 'age', 'sex'], axis=1)
        feature_vals = list(X.columns)

        return X, Y, feature_vals

    def find_biomarkers_brain_resilience(self, X, Y, features):
        clf_imp_features = dict()
        vulresMinus_native = {}
        final_list_train = []
        final_list_test = []
        X_iter = pd.DataFrame()
        region1_list = []
        region2_list = []
        all_conn = 62 * 62
        train_acc = 0
        test_acc = 0
        val_acc = 0

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        clf, score, conf_matrix, pred = get_classification(X_train, Y_train, "logistic")
        print('Logistic Regression Accuracy based on 80% data: ', score)
        importance = clf.coef_[0]

        # Summarize feature importance
        for i, v in enumerate(importance):
            clf_imp_features[features[i]] = v
        # plot feature importance
        # plt.hist(importance, bins=100)
        # plt.show()

        imp_feat = dict(sorted(clf_imp_features.items(), key=lambda item: item[1], reverse=True))

        for key1, val1 in self.meanDiff_vulresMinus_native.iterrows():
            for key2, val2 in val1.iteritems():
                vulresMinus_native[(key1, key2)] = val2

        vulresMinus_native_negative = sorted(vulresMinus_native.items(), key=operator.itemgetter(1))

        df = pd.DataFrame(vulresMinus_native_negative, columns=['Connectomes', 'VulMinusRes_Group'])

        df['feature_importance'] = ""

        for key, val in imp_feat.items():
            for i in range(0, len(df['Connectomes'])):
                if key == df['Connectomes'][i]:
                    df['feature_importance'][i] = val

        for i in range(0, len(df)):
            if df['feature_importance'][i] == "":
                df.drop(index=i, inplace=True)
        # df.to_csv('connectomes_importance.csv', encoding='utf-8')

        df_feat_imp_abs = df.copy()
        df_feat_imp_abs['feature_importance'] = df_feat_imp_abs['feature_importance'].abs()
        df_feat_imp_abs = df_feat_imp_abs.sort_values(by=['feature_importance'], ascending=False)

        for i in df_feat_imp_abs[:30].index:
            region1_list.append(df_feat_imp_abs['Connectomes'][i][0])
            region2_list.append(df_feat_imp_abs['Connectomes'][i][1])

        for i in range(0, len(region1_list)):
            try:
                final_list_train.append(X_train[[(region1_list[i], region2_list[i])]])
                final_list_test.append(X_test[[(region1_list[i], region2_list[i])]])
            except:
                final_list_train.append(X_train[[(region2_list[i], region1_list[i])]])
                final_list_test.append(X_test[[(region2_list[i], region1_list[i])]])

        X_iter_train = pd.concat(final_list_train, axis=1)
        X_iter_test = pd.concat(final_list_test, axis=1)

        # print(X_iter_train.shape)
        # print(Y_train.shape)
        # print(X_iter_test.shape)
        # print(Y_test.shape)

        X_train, X_val, Y_train, Y_val = train_test_split(X_iter_train, Y_train, test_size=0.2, random_state=0)

        abs_metrics = find_best_features(X_train, Y_train, X_val, Y_val, X_iter_test, Y_test,
                                         train_acc, test_acc, val_acc)

        return abs_metrics
