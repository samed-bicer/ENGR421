################################################
####                                        ####
####  Mehmet Samed Bicer - 0050464          ####
####  ENGR421 Homework-08                   ####
####  Koc University, Istanbul - 5-Jan-20   ####
####                                        ####
################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split  # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


# define ROC curve method
def plot_roc_curve(y_test, y_pred, name):
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test.values, y_pred)
    roc_auc = auc(fpr, tpr)

    print('AUROC = %0.2f' % roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve - ' + name)
    plt.legend(loc="lower right")
    plt.show()


def conf_matrix(y_test, y_pred, name):
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('Confusion matrix:\n', conf_mat)

    # labels = ['Class 0', 'Class 1']
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    # fig.colorbar(cax)
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)
    # plt.title('Conf. Matrix - ' + name)
    # plt.xlabel('Predicted')
    # plt.ylabel('Expected')
    # plt.show()


def preprocess():
    print('reading dataset...')
    x_train = pd.read_csv('hw08_training_data.csv')
    x_test = pd.read_csv('hw08_test_data.csv')
    y_train = pd.read_csv('hw08_training_label.csv')

    print('preprocessing...')
    numeric_data = x_train.select_dtypes(include=[np.number])
    non_numeric_data = x_train.select_dtypes(exclude=[np.number])
    print("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],
                                                                                 non_numeric_data.shape[1]))

    numeric_data_test = x_test.select_dtypes(include=[np.number])
    non_numeric_data_test = x_test.select_dtypes(exclude=[np.number])
    print("There are {} numeric and {} categorical columns in test data".format(numeric_data_test.shape[1],
                                                                                non_numeric_data_test.shape[1]))

    # Fill na values with column mean values
    numeric_data = numeric_data.fillna(numeric_data.mean())
    numeric_data_test = numeric_data_test.fillna(numeric_data_test.mean())

    y_train = y_train.fillna(0)

    # Scale numeric values
    ss = StandardScaler()
    numeric_data.iloc[:, 1:167] = pd.DataFrame(ss.fit_transform(numeric_data.iloc[:, 1:167]),
                                               columns=numeric_data.iloc[:, 1:167].columns)
    numeric_data_test.iloc[:, 1:167] = pd.DataFrame(ss.fit_transform(numeric_data_test.iloc[:, 1:167]),
                                                    columns=numeric_data_test.iloc[:, 1:167].columns)

    # Encode non-numeric columns
    non_numeric_data = pd.get_dummies(non_numeric_data, columns=non_numeric_data.columns, drop_first=True)
    non_numeric_data_test = pd.get_dummies(non_numeric_data_test, columns=non_numeric_data_test.columns,
                                           drop_first=True)

    x_training = pd.concat([numeric_data, non_numeric_data], axis=1)
    x_testing = pd.concat([numeric_data_test, non_numeric_data_test], axis=1)

    svd = TruncatedSVD(n_components=100, n_iter=20, random_state=421)

    x_train_fitted = pd.concat([pd.DataFrame(svd.fit_transform(x_training.iloc[:, 1:657])), y_train.iloc[:, 1:7]],
                               axis=1)
    x_test_fitted = pd.DataFrame(svd.fit_transform(x_testing.iloc[:, 1:657]))
    # for the column order
    x_test_fitted = x_test_fitted[x_train_fitted.iloc[:, 0:-6].columns]
    x_test_fitted = pd.concat([pd.DataFrame(x_test['ID']), x_test_fitted], axis=1)

    print('dataset preprocessed')
    return x_train_fitted, x_test_fitted


def xgBoostTrain(x_train, x_test, y_train, y_test):
    D_train = xgb.DMatrix(data=x_train, label=y_train)
    D_test = xgb.DMatrix(data=x_test)

    param = {'objective': 'binary:logistic'}
    steps = 20  # The number of training iterations

    # model training
    model = xgb.train(param, D_train, steps)

    # prediction of test values
    xgb_pred = model.predict(D_test)

    plot_roc_curve(y_test, xgb_pred, 'XGBoost on validation set')

    target = np.zeros(len(xgb_pred))
    target[xgb_pred > 0.5] = 1

    conf_matrix(y_test, target, 'Conf. matrix for validation set')

    return model


def dataOverSampling(df_class_0, df_class_1):
    print('Data over sampling...')
    count_class_0 = len(df_class_0)

    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    return df_test_over


if __name__ == '__main__':
    x_train, x_test = preprocess()

    ####################
    ####  TARGET_1  ####
    ####################

    print('-----TARGET_1-----')
    df_class_target1_0 = x_train[x_train['TARGET_1'] == 0]
    df_class_target1_1 = x_train[x_train['TARGET_1'] == 1]

    df_test_target1_over = dataOverSampling(df_class_target1_0, df_class_target1_1)

    X_train_target1, X_valid_target1, Y_train_target1, Y_valid_target1 = train_test_split(
        df_test_target1_over.iloc[:, 0:-6], df_test_target1_over['TARGET_1'], test_size=0.2, random_state=421)

    model_target1 = xgBoostTrain(X_train_target1, X_valid_target1, Y_train_target1, Y_valid_target1)
    xgb_pred_target1 = model_target1.predict(xgb.DMatrix(data=x_test.iloc[:, 1:]))

    all_predictions = pd.concat([pd.DataFrame(x_test['ID']), pd.DataFrame(data=xgb_pred_target1, columns=['TARGET_1'])],
                                axis=1)

    print('TARGET_1 probabilities are calculated.')

    ####################
    ####  TARGET_2  ####
    ####################

    print('-----TARGET_2-----')
    df_class_target2_0 = x_train[x_train['TARGET_2'] == 0]
    df_class_target2_1 = x_train[x_train['TARGET_2'] == 1]

    df_test_target2_over = dataOverSampling(df_class_target2_0, df_class_target2_1)

    X_train_target2, X_valid_target2, Y_train_target2, Y_valid_target2 = train_test_split(
        df_test_target2_over.iloc[:, 0:-6], df_test_target2_over['TARGET_2'], test_size=0.2, random_state=421)

    model_target2 = xgBoostTrain(X_train_target2, X_valid_target2, Y_train_target2, Y_valid_target2)
    xgb_pred_target2 = model_target2.predict(xgb.DMatrix(data=x_test.iloc[:, 1:]))

    all_predictions = pd.concat([all_predictions, pd.DataFrame(data=xgb_pred_target2, columns=['TARGET_2'])],
                                axis=1)

    print('TARGET_2 probabilities are calculated.')

    ####################
    ####  TARGET_3  ####
    ####################

    print('-----TARGET_3-----')
    df_class_target3_0 = x_train[x_train['TARGET_3'] == 0]
    df_class_target3_1 = x_train[x_train['TARGET_3'] == 1]

    df_test_target3_over = dataOverSampling(df_class_target3_0, df_class_target3_1)

    X_train_target3, X_valid_target3, Y_train_target3, Y_valid_target3 = train_test_split(
        df_test_target3_over.iloc[:, 0:-6], df_test_target3_over['TARGET_3'], test_size=0.2, random_state=421)

    model_target3 = xgBoostTrain(X_train_target3, X_valid_target3, Y_train_target3, Y_valid_target3)
    xgb_pred_target3 = model_target3.predict(xgb.DMatrix(data=x_test.iloc[:, 1:]))

    all_predictions = pd.concat([all_predictions, pd.DataFrame(data=xgb_pred_target3, columns=['TARGET_3'])],
                                axis=1)

    print('TARGET_3 probabilities are calculated.')

    ####################
    ####  TARGET_4  ####
    ####################

    print('-----TARGET_4-----')
    df_class_target4_0 = x_train[x_train['TARGET_4'] == 0]
    df_class_target4_1 = x_train[x_train['TARGET_4'] == 1]

    df_test_target4_over = dataOverSampling(df_class_target4_0, df_class_target4_1)

    X_train_target4, X_valid_target4, Y_train_target4, Y_valid_target4 = train_test_split(
        df_test_target4_over.iloc[:, 0:-6], df_test_target4_over['TARGET_4'], test_size=0.2, random_state=421)

    model_target4 = xgBoostTrain(X_train_target4, X_valid_target4, Y_train_target4, Y_valid_target4)
    xgb_pred_target4 = model_target4.predict(xgb.DMatrix(data=x_test.iloc[:, 1:]))

    all_predictions = pd.concat([all_predictions, pd.DataFrame(data=xgb_pred_target4, columns=['TARGET_4'])],
                                axis=1)

    print('TARGET_4 probabilities are calculated.')

    ####################
    ####  TARGET_5  ####
    ####################

    print('-----TARGET_5-----')
    df_class_target5_0 = x_train[x_train['TARGET_5'] == 0]
    df_class_target5_1 = x_train[x_train['TARGET_5'] == 1]

    df_test_target5_over = dataOverSampling(df_class_target5_0, df_class_target5_1)

    X_train_target5, X_valid_target5, Y_train_target5, Y_valid_target5 = train_test_split(
        df_test_target5_over.iloc[:, 0:-6], df_test_target5_over['TARGET_5'], test_size=0.2, random_state=421)

    model_target5 = xgBoostTrain(X_train_target5, X_valid_target5, Y_train_target5, Y_valid_target5)
    xgb_pred_target5 = model_target5.predict(xgb.DMatrix(data=x_test.iloc[:, 1:]))

    all_predictions = pd.concat([all_predictions, pd.DataFrame(data=xgb_pred_target5, columns=['TARGET_5'])],
                                axis=1)

    print('TARGET_5 probabilities are calculated.')

    ####################
    ####  TARGET_6  ####
    ####################

    print('-----TARGET_6-----')

    df_class_target6_0 = x_train[x_train['TARGET_6'] == 0]
    df_class_target6_1 = x_train[x_train['TARGET_6'] == 1]

    df_test_target6_over = dataOverSampling(df_class_target6_0, df_class_target6_1)

    X_train_target6, X_valid_target6, Y_train_target6, Y_valid_target6 = train_test_split(
        df_test_target6_over.iloc[:, 0:-6], df_test_target6_over['TARGET_6'], test_size=0.2, random_state=421)

    model_target6 = xgBoostTrain(X_train_target6, X_valid_target6, Y_train_target6, Y_valid_target6)
    xgb_pred_target6 = model_target6.predict(xgb.DMatrix(data=x_test.iloc[:, 1:]))

    all_predictions = pd.concat([all_predictions, pd.DataFrame(data=xgb_pred_target6, columns=['TARGET_6'])],
                                axis=1)
    print('TARGET_6 probabilities are calculated.')

    export_csv = all_predictions.to_csv(r'hw08_test_predictions.csv', index=None, header=True)

    print('all target values are written into hw08_test_predictions.csv')
