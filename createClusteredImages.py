import numpy as np
import pandas as pd
from scipy.io import loadmat
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nibabel as nib
from nilearn import plotting

# from utils import convertMatToDataframe
from utils import add_labels_to_df


def convertMatToDataframe(mat_file):
    colNames = mat_file['names']

    listNames = []

    for i in range(0, 62):
        listNames.append(''.join(colNames[0][i]))

    # print(listNames)
    globals()['finalList'] = listNames

    mData = mat_file['Z']
    mData = np.delete(mData, np.s_[62:], axis=1)
    df = pd.DataFrame(mData)
    df.columns = listNames
    df.index = listNames
    return df


if __name__ == "__main__":

    vulnerableList = []
    resilientList = []
    controlList = []

    cluster_mean_0 = []
    cluster_mean_1 = []
    cluster_mean_2 = []
    cluster_mean_3 = []
    cluster_mean_4 = []
    cluster_mean_5 = []
    cluster_mean_6 = []
    cluster_mean_7 = []

    finalList = []

    number_of_clusters = 5

    flag = 0

    numpy_native_3d = np.zeros(shape=(62, 62))
    vul_numpy_native_3d = np.zeros(shape=(62, 62))
    res_numpy_native_3d = np.zeros(shape=(62, 62))
    control_numpy_native_3d = np.zeros(shape=(62, 62))

    numpy_mni_3d = np.zeros(shape=(62, 62))
    vul_numpy_mni_3d = np.zeros(shape=(62, 62))
    res_numpy_mni_3d = np.zeros(shape=(62, 62))
    control_numpy_mni_3d = np.zeros(shape=(62, 62))

    subjects = ['Subject001', 'Subject002', 'Subject003', 'Subject004', 'Subject005', 'Subject006', 'Subject007',
                'Subject008', 'Subject009', 'Subject010', 'Subject011', 'Subject012', 'Subject013', 'Subject014',
                'Subject015', 'Subject016', 'Subject017', 'Subject018', 'Subject019', 'Subject020', 'Subject021',
                'Subject022', 'Subject023', 'Subject024', 'Subject025', 'Subject026', 'Subject027', 'Subject028',
                'Subject029', 'Subject030', 'Subject031', 'Subject032', 'Subject033', 'Subject034', 'Subject035',
                'Subject036', 'Subject037', 'Subject038', 'Subject039', 'Subject040', 'Subject041', 'Subject042',
                'Subject043', 'Subject044', 'Subject045', 'Subject046']

    # Z stands for \\rdcemea-t3.storage.philips.com\011497_mri_eindhoven_ux\Resilience_project_with_LUMC\
    rootDirPath = pathlib.Path(r'Z:\code')
    subjectList = pd.read_csv(r'Z:\data\bids\participants_updated.tsv', sep='\t')
    path_to_atlas = r'Z:\data\bids\bids_native_final\sub-resi005\ses-1\\anat\sub-resi005_ses-1_T1w_vwbs_atlas.nii'

    for index, row in subjectList.iterrows():
        # print(row)
        if row['diagnosis'] == 'vulnerable':
            vulnerableList.append(row['subject_id'])
        elif row['diagnosis'] == 'resilient':
            resilientList.append(row['subject_id'])
        else:
            controlList.append(row['subject_id'])

    for subject in subjects:
        mat_file_native = loadmat(pathlib.Path(rootDirPath / 'conn_project_native' /
                                               'results' / 'firstlevel' / 'SBC_01'
                                               / ('resultsROI_' + subject + '_Condition001.mat')))

        mat_file_mni = loadmat(pathlib.Path(rootDirPath / 'matlab' / 'conn_LUMC_dataset1' /
                                            'results' / 'firstlevel' / 'SBC_01'
                                            / ('resultsROI_' + subject + '_Condition001.mat')))

        df_file_native = convertMatToDataframe(mat_file_native)
        numpy_native = df_file_native.to_numpy()

        df_file_mni = convertMatToDataframe(mat_file_mni)
        numpy_mni = df_file_mni.to_numpy()

        # Native space
        if (numpy_native_3d == 0).all():
            print(' Native Empty')
            numpy_native_3d = numpy_native
        else:
            numpy_native_3d = np.dstack([numpy_native_3d, numpy_native])

        if subject in vulnerableList:
            if (vul_numpy_native_3d == 0).all():
                print('Native Vulnerable Empty')
                vul_numpy_native_3d = numpy_native
            else:
                vul_numpy_native_3d = np.dstack([vul_numpy_native_3d, numpy_native])
        elif subject in resilientList:
            if (res_numpy_native_3d == 0).all():
                print('Native Resilient Empty')
                res_numpy_native_3d = numpy_native
            else:
                res_numpy_native_3d = np.dstack([res_numpy_native_3d, numpy_native])
        else:
            if (control_numpy_native_3d == 0).all():
                print('Native Control Group Empty')
                control_numpy_native_3d = numpy_native
            else:
                control_numpy_native_3d = np.dstack([control_numpy_native_3d, numpy_native])

        # MNI Space
        if (numpy_mni_3d == 0).all():
            print('MNI Empty')
            numpy_mni_3d = numpy_mni
        else:
            numpy_mni_3d = np.dstack([numpy_mni_3d, numpy_mni])

        if subject in vulnerableList:
            if (vul_numpy_mni_3d == 0).all():
                print(' MNI Vulnerable Empty')
                vul_numpy_mni_3d = numpy_mni
            else:
                vul_numpy_mni_3d = np.dstack([vul_numpy_mni_3d, numpy_mni])
        elif subject in resilientList:
            if (res_numpy_mni_3d == 0).all():
                print(' MNI Resilient Empty')
                res_numpy_mni_3d = numpy_mni
            else:
                res_numpy_mni_3d = np.dstack([res_numpy_mni_3d, numpy_mni])
        else:
            if (control_numpy_mni_3d == 0).all():
                print('MNI Control Group Empty')
                control_numpy_mni_3d = numpy_mni
            else:
                control_numpy_mni_3d = np.dstack([control_numpy_mni_3d, numpy_mni])

    meanMatrix_native = np.mean(numpy_native_3d, axis=2)
    vul_meanMatrix_native = np.mean(vul_numpy_native_3d, axis=2)
    res_meanMatrix_native = np.mean(res_numpy_native_3d, axis=2)
    ctrl_meanMatrix_native = np.mean(control_numpy_native_3d, axis=2)

    meanMatrix_mni = np.mean(numpy_mni_3d, axis=2)
    vul_meanMatrix_mni = np.mean(vul_numpy_mni_3d, axis=2)
    res_meanMatrix_mni = np.mean(res_numpy_mni_3d, axis=2)
    ctrl_meanMatrix_mni = np.mean(control_numpy_mni_3d, axis=2)

    meanDF_native = pd.DataFrame(meanMatrix_native)
    meanDF_native = add_labels_to_df(meanDF_native)

    meanDF_mni = pd.DataFrame(meanMatrix_mni)
    meanDF_mni = add_labels_to_df(meanDF_mni)

    meanDF_native_numpy = meanDF_native.to_numpy()
    # print(np.nanmax(meanDF_native_numpy))
    meanDF_native_numpy = np.nan_to_num(meanDF_native_numpy, nan=np.nanmax(meanDF_native_numpy))
    # print(meanDF_native_numpy.shape)

    meanDF_mni_numpy = meanDF_mni.to_numpy()
    meanDF_mni_numpy = np.nan_to_num(meanDF_mni_numpy, nan=np.nanmax(meanDF_native_numpy))

    plt.rcParams["figure.figsize"] = (3, 3)  # see a tutorial on clustermap and explain the parameters.

    sns.clustermap(meanDF_native_numpy, center=0)

    rowIndices = sns.clustermap(meanDF_native_numpy).dendrogram_row.reordered_ind
    colIndices = sns.clustermap(meanDF_native_numpy).dendrogram_col.reordered_ind

    # Reorder the connectivity matrices with differences between groups. Find out which groups show higher differences.

    meanDF_native_df_final = pd.DataFrame(meanDF_native_numpy)
    meanDF_native_df_final = meanDF_native_df_final.reindex(rowIndices, columns=colIndices)

    # sub_list = np.split(rowIndices, np.arange(8, len(rowIndices), 8), axis=0)
    # sub_list = [[18, 49, 10, 41, 2, 33, 8, 39, 4, 35],
    #             [26, 57, 14, 45, 19, 50, 21, 52],
    #             [53, 22, 7, 38, 5, 36],
    #             [23, 54, 13, 44, 11, 42, 3, 34],
    #             [20, 51, 30, 61, 29, 60, 0, 31, 46, 59],
    #             [27, 58, 12, 43, 6, 37],
    #             [1, 25, 32, 56, 9, 40, 16, 47, 48, 15, 17, 28, 24, 55]
    #             ]

    # sub_list = [[18, 49, 10, 41, 2, 33, 8, 39, 4, 35],
    #             [26, 57, 14, 45, 19, 50, 21, 52],
    #             [53, 22, 7, 38, 5, 36],
    #             [23, 54, 13, 44, 11, 42, 3, 34],
    #             [20, 51, 30, 61, 29, 60, 0, 31, 46, 59],
    #             [27, 58, 12, 43, 6, 37],
    #             [1, 25, 32, 56],
    #             [9, 40, 16, 47, 48, 15, 17, 28, 24, 55]
    #             ]

    # sub_list = [[18, 49, 10, 41, 2, 33, 8, 39, 4, 35],
    #             [26, 57, 14, 45, 19, 50, 21, 52],
    #             [53, 22, 7, 38, 5, 36],
    #             [23, 54, 13, 44, 11, 42, 3, 34],
    #             [20, 51, 30, 61, 29, 60, 0, 31, 46, 59],
    #             [27, 58, 12, 43, 6, 37, 1, 25, 32, 56, 9, 40, 16, 47, 48, 15, 17, 28, 24, 55]
    #             ]

    sub_list = [[18, 49, 10, 41, 2, 33, 8, 39, 4, 35],
                [26, 57, 14, 45, 19, 50, 21, 52],
                [53, 22, 7, 38, 5, 36, 23, 54, 13, 44, 11, 42, 3, 34],
                [20, 51, 30, 61, 29, 60, 0, 31, 46, 59],
                [27, 58, 12, 43, 6, 37, 1, 25, 32, 56, 9, 40, 16, 47, 48, 15, 17, 28, 24, 55]
                ]
    # sub_list_columns = np.split(colIndices, np.arange(8, len(colIndices), 8), axis=0)
    print('finalList:: ', finalList)
    for region in finalList:
        val = int(re.search(r'\d+', region).group())
        # print(val)

        for i in range(0, number_of_clusters):
            if (val - 1) in sub_list[i]:
                # print(str(val) + " in " + str(sub_list[i]) + " under " + str('cluster_mean_{}'.format(i)))
                (globals()['cluster_mean_{}'.format(i)]).append(val)

    img = nib.load(path_to_atlas)
    img_data = img.get_fdata()
    new_img_data = img.get_fdata()

    print('Cluster 1: ', cluster_mean_0)
    print('Cluster 2: ', cluster_mean_1)
    print('Cluster 3: ', cluster_mean_2)
    print('Cluster 4: ', cluster_mean_3)
    print('Cluster 5: ', cluster_mean_4)
    print('Cluster 6: ', cluster_mean_5)
    print('Cluster 7: ', cluster_mean_6)
    print('Cluster 8: ', cluster_mean_7)

    # print('img_data:: ', img_data)
    print('Finally entering the dreaded zone')
    for x in range(0, 256):
        # print('x: ', x)
        for y in range(0, 256):
            # print('y: ', y)
            for z in range(0, 140):
                if img_data[x][y][z] > 0:
                    # print('Next value:: ', img_data[x][y][z])
                    for i in range(0, number_of_clusters):
                        if int(img_data[x][y][z]) in (globals()['cluster_mean_{}'.format(i)]):
                            # print(img_data[x][y][z], ' ', (globals()['cluster_mean_{}'.format(i)]))
                            print(img_data[x][y][z], 'is now in cluster ', i + 1)
                            img_data[x][y][z] = i + 1
                            break

    new_clustered_img = nib.Nifti1Image(img_data, img.affine)
    # Change the location here - to be changed using pathlib
    nib.save(new_clustered_img, r'Z:\data\bids\bids_native_final\sub-resi005\ses-1\anat\sub-resi005_ses'
                                r'-1_T1w_vwbs_atlas_clustered_5.nii')

    plotting.plot_roi(img, new_clustered_img)
    print('Done')
