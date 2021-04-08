import pandas as pd
import numpy as np
import shutil
import pathlib
import time
import nibabel as nb
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def convertMatlabToDataframe(mat_file):
    columnNames = mat_file['names']
    listNames = []

    for i in range(0, 62):
        listNames.append(''.join(columnNames[0][i]))

    globals()['finalList'] = listNames

    mData = mat_file['Z']
    mData = np.delete(mData, np.s_[62:], axis=1)
    df = pd.DataFrame(mData)
    df.columns = listNames
    df.index = listNames
    return df


def add_labels_to_df(df):
    ordered_region_names = []
    labelList = pd.read_csv(r'A:\data\bids\bids_native_final\sub_ses-1_T1w_vwbs_atlas.csv', sep=',')

    for key1, val1 in df.iterrows():
        # print(key1)
        ordered_region_names.append(labelList['brain_regions'][key1])

    df.columns = ordered_region_names
    df.index = ordered_region_names
    return df


def get_coregistered_atlas_native_space(rootDir, t1_file, vwbs_file, vwbs_file_txt,
                                        dwi_folder,
                                        bold_func_file_nii, bold_func_file_json,
                                        hires_func_file_nii):
    """
    Creates a new file where the vwbs images are binarized
    Creates new folders in the format required for conn processing in a new location
    and co-registered in accordance to the T1-file
    :param rootDir: root of the ROIs
    :param t1_file: The T1 brain segmentation anatomical image location
    :param vwbs_file: The voxel wise brain segmentation anatomical image location
    :return: None
    """

    # print(vwbs_file.parent)
    # print(vwbs_file.stem)
    # print(vwbs_file.suffix)

    # Load the T1-subject file from the t1_file location
    subject_t1_file = nb.load(t1_file)

    # Load the voxel wise brain segmentation file from the vwbs_file location
    subject_space_atlas_native_img = nb.load(vwbs_file)

    # print(subject_space_atlas_native_img.shape)

    # Get the data values of the subject in numpy format
    subject_space_atlas_native_data = subject_space_atlas_native_img.get_fdata()

    # Binarize the Brain ROI values
    subject_space_atlas_native_data = np.rint(subject_space_atlas_native_data)

    # Localize the brain regions from 0 to 62 ( putting regions more than 62
    # to background values i.e. 0 )
    for x in range(0, 256):
        # print('x: ', x)
        for y in range(0, 256):
            # print('y: ', y)
            for z in range(0, 139):
                if subject_space_atlas_native_data[x][y][z] > 62:
                    subject_space_atlas_native_data[x][y][z] = 0

    print(vwbs_file.stem, "Completed")

    # Convert the numpy values to niftii image
    subject_space_atlas_native_final_img = nb.Nifti1Image(
        subject_space_atlas_native_data, subject_t1_file.affine)

    # Create new directories due to copy restrictions
    subjectDir = rootDir / vwbs_file.parent.parent.parent.stem
    subjectDir.mkdir()
    sessionDir = subjectDir / vwbs_file.parent.parent.stem
    sessionDir.mkdir()
    anatDir = sessionDir / vwbs_file.parent.stem
    anatDir.mkdir()
    funcDir = sessionDir / bold_func_file_nii.parent.stem
    funcDir.mkdir()

    # Copy the T1-file to the new location
    try:
        shutil.copy(t1_file, anatDir)
        print(t1_file, "copied")
    except:
        print(t1_file, "not found")

    # Copy the vwbs-file to the new location
    try:
        shutil.copy(vwbs_file, anatDir)
        print(vwbs_file, "copied")
    except:
        print(vwbs_file, "not found")

    # Copy the vwbs-file to the new location
    try:
        shutil.copy(vwbs_file_txt, anatDir)
        print(vwbs_file_txt, "copied")
    except:
        print(vwbs_file_txt, "not found")

    # Saving the new suject specific ROI atlas in the new location
    nb.save(subject_space_atlas_native_final_img,
            pathlib.Path(anatDir, vwbs_file.stem + "_atlas" + vwbs_file.suffix))
    print((vwbs_file.stem + "_atlas" + vwbs_file.suffix), " created")

    # Copy the contents of the DWI folder
    try:
        shutil.copytree(dwi_folder, (sessionDir / "dwi"))
        print(dwi_folder, "copied")
    except:
        print(dwi_folder, "not found")

    # Copy the BOLD-nii to the new location
    try:
        shutil.copy(bold_func_file_nii, funcDir)
        print(bold_func_file_nii, "copied")
    except:
        print(bold_func_file_nii, "not found")

    # Copy the BOLD-json to the new location
    try:
        shutil.copy(bold_func_file_json, funcDir)
        print(bold_func_file_json, "copied")
    except:
        print(bold_func_file_json, "not found")

    # Copy the hires-nii to the new location
    try:
        shutil.copy(hires_func_file_nii, funcDir)
        print(hires_func_file_nii, "copied")
    except:
        print(hires_func_file_nii, "not found")


def get_classification(X, Y, model):
    global predictions, scores, cm
    best_score = 0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    if model == 'logistic':
        clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
        for fold in range(1, 10):
            cv = KFold(n_splits=fold + 1)
            scores = cross_val_score(clf, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                predictions = cross_val_predict(clf, X, Y, cv=cv)
                cm = metrics.confusion_matrix(Y, predictions)
        # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        return clf, best_score, cm, predictions


def find_best_features(X_iter, Y_train, X_val, Y_val, X_test, Y_test, train_acc, test_acc, val_acc, best_features):
    df_metrics = pd.DataFrame(
        columns=['Number of features', 'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy', 'Best Epoch'])
    k = 0

    for i in range(5, 6):  # len(X_iter.columns)):
        X_train_final = X_iter.iloc[:, : i]
        X_test_final = X_test.iloc[:, : i]
        X_val_final = X_val.iloc[:, : i]
        # test_acc, best_features, train_acc, val_acc =
        train_acc, val_acc, test_acc, best_epoch = get_mlp_classification(X_train_final, Y_train, X_val_final, Y_val,
                                                              X_test_final, Y_test, train_acc,
                                                              test_acc, val_acc)
        df_metrics.loc[k, 'Number of features'] = i
        df_metrics.loc[k, 'Training Accuracy'] = train_acc
        df_metrics.loc[k, 'Validation Accuracy'] = val_acc
        df_metrics.loc[k, 'Testing Accuracy'] = test_acc
        df_metrics.loc[k, 'Best epoch'] = best_epoch
        k += 1

    return df_metrics


def get_mlp_classification(X_train, Y_train, X_val, Y_val, X_test, Y_test, train_accuracy, test_accuracy, val_accuracy,
                           best_features):
    n_features = X_train.shape[1]
    n_batch = len(X_train)
    print(n_features)

    best_val_acc = 0
    best_train_acc = 0
    best_epoch = 0

    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=0.1)
    # Instantiate a loss function.
    # loss_fn = keras.losses.binary_crossentropy(from_logits=True)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(n_batch)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    val_dataset = val_dataset.batch(n_batch)

    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    inputs = keras.Input(shape=(n_features,), name="adj_mat")
    x1 = tf.keras.layers.Dense(1891, activation="relu")(inputs)
    x1_Drouput = tf.keras.layers.Dropout(0.4)(x1)
    x2 = tf.keras.layers.Dense(512, activation="relu")(x1_Drouput)
    x2_Drouput = tf.keras.layers.Dropout(0.4)(x2)
    x3 = tf.keras.layers.Dense(256, activation="relu")(x2_Drouput)
    x3_Drouput = tf.keras.layers.Dropout(0.4)(x3)
    x4 = tf.keras.layers.Dense(64, activation="relu")(x3_Drouput)
    x4_Drouput = tf.keras.layers.Dropout(0.4)(x4)
    outputs = tf.keras.layers.Dense(2, name="predictions")(x4_Drouput)
    model = keras.Model(inputs=inputs, outputs=outputs)

    epochs = 500
    for epoch in range(epochs):
        print("\nStart of epoch %d for %d features" % (epoch, n_features))
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch
                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every batch.
            # print("Training loss at step %d: %.4f" % (step, float(loss_value)))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))

        if float(val_acc) > best_val_acc:
            best_val_acc = float(val_acc)
            best_train_acc = float(train_acc)
            best_epoch = epoch
            model.save(pathlib.Path(r'C:\Users\320106459\Desktop\Project\Models\best_model'))
            print('Saving best model')

    # model when the val acc is the highest and plot the epoch vs performance
    # no. of epochs(when stopped) vs no. of features

    # evaluate the model
    best_model = keras.models.load_model(pathlib.Path(r'C:\Users\320106459\Desktop\Project\Models\best_model'))
    best_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    loss, acc = best_model.evaluate(X_test, Y_test)
    print('Test Accuracy: %.3f' % acc, ' for ', n_features, ' features')
    best_test_acc = acc

    return best_train_acc, best_val_acc, best_test_acc, best_epoch
