import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split  # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler


# define ROC curve method
def plot_roc_curve(y_test, y_pred, name):
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test.values, y_pred)
    roc_auc = auc(fpr, tpr)

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


def preprocess(index):
    x_train = pd.read_csv('hw07_target' + str(index) + '_training_data.csv')
    x_test = pd.read_csv('hw07_target' + str(index) + '_test_data.csv')
    y_train = pd.read_csv('hw07_target' + str(index) + '_training_label.csv')

    # missing values for columns which has more null elements than non-nulls
    miss = x_train.isnull().sum() / len(x_train)
    miss = miss[miss > 0.5]
    miss.sort_values(inplace=True)

    # visualising missing values
    # plot the missing value count
    # miss = miss.to_frame()
    # miss.columns = ['count']
    # miss['Name'] = miss.index
    # sns.set(style="whitegrid", color_codes=True)
    # sns.barplot(x='Name', y='count', data=miss)
    # plt.xticks(rotation=90)
    # plt.show()

    # drop these columns
    x_train = x_train.drop(columns=miss.index)
    x_test = x_test.drop(columns=miss.index)

    # drop ID column
    # del x_train['ID']

    numeric_data = x_train.select_dtypes(include=[np.number])
    non_numeric_data = x_train.select_dtypes(exclude=[np.number])

    numeric_data_test = x_test.select_dtypes(include=[np.number])
    non_numeric_data_test = x_test.select_dtypes(exclude=[np.number])

    print("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],
                                                                                 non_numeric_data.shape[1]))
    # Fill na values with column mean values
    numeric_data = numeric_data.fillna(numeric_data.mean())
    numeric_data_test = numeric_data_test.fillna(numeric_data_test.mean())

    # Scale numeric values
    ss = StandardScaler()
    numeric_data = pd.DataFrame(ss.fit_transform(numeric_data))
    numeric_data_test = pd.DataFrame(ss.fit_transform(numeric_data_test))

    # Encode non-numeric columns
    non_numeric_data = pd.get_dummies(non_numeric_data, columns=non_numeric_data.columns, drop_first=True)
    non_numeric_data_test = pd.get_dummies(non_numeric_data_test, columns=non_numeric_data_test.columns,
                                           drop_first=True)

    # Dataset to be split
    x_training = pd.concat([non_numeric_data, numeric_data], axis=1)
    x_real_test = pd.concat([non_numeric_data_test, numeric_data_test], axis=1)

    # X_train, X_test, Y_train, Y_test for training and validation, Real test set

    X_train, X_test, Y_train, Y_test = train_test_split(x_training, y_train['TARGET'], test_size=0.2, random_state=2020)

    return X_train, X_test, Y_train, Y_test, x_real_test


def bestFeatures(x_train, y_train, x_test):
    # best k=50 features of the dataset according to F values of features
    bestfeatures = SelectKBest(score_func=f_regression, k=50)
    fit = bestfeatures.fit(x_train, y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_train.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    bestColumns = featureScores.nlargest(50, 'Score')['Specs'].values

    X_train = x_train[bestColumns]
    X_test = x_test[bestColumns]

    return X_train, X_test


def KerasRegression(x_train, y_train, x_test, y_test, x_real_test, i):
    # create Model
    # define base model
    def base_model():
        model = Sequential()
        model.add(Dense(35, input_dim=len(x_train.columns), activation="relu", kernel_initializer="normal"))
        model.add(Dense(16, activation="relu", kernel_initializer="normal"))
        model.add(Dense(1, kernel_initializer="normal"))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    keras_label = y_train.as_matrix()
    clf = KerasRegressor(build_fn=base_model, nb_epoch=1000, batch_size=5, verbose=0)
    clf.fit(x_train, keras_label)

    # make predictions
    keras_pred = clf.predict(x_test)
    keras_pred = np.exp(keras_pred)

    plot_roc_curve(y_test, keras_pred, 'Keras Reg. Target: ' + str(i + 1))

    #keras_real_pred = clf.predict(x_real_test)
    #keras_real_pred = np.exp(keras_real_pred)

    return keras_pred


def LassoRegression(x_train, y_train, x_test, y_test, x_real_test, i):
    regr = Lasso(alpha=0.00099, max_iter=50000)
    regr.fit(x_train, y_train)

    # run prediction on the training set to get a rough idea of how well it does
    lasso_pred = regr.predict(x_test)

    plot_roc_curve(y_test, lasso_pred, 'Lasso Reg. Target: ' + str(i + 1))

    #lasso_real_pred = regr.predict(x_real_test)

    return lasso_pred


def xgBoostTrain(x_train, y_train, x_test, y_test, x_real_test, i):
    D_train = xgb.DMatrix(data=x_train, label=y_train)
    D_test = xgb.DMatrix(data=x_test, label=y_test)

    D_real_test = xgb.DMatrix(data=x_real_test)

    param = {'objective': 'binary:logistic'}
    steps = 20  # The number of training iterations

    # model training
    model = xgb.train(param, D_train, steps)

    # prediction of test values
    xgb_pred = model.predict(D_test)

    plot_roc_curve(y_test, xgb_pred, 'xgBoost Target: ' + str(i + 1))

    #xgb_real_pred = model.predict(D_real_test)

    return xgb_pred


if __name__ == '__main__':

    for i in range(3):
        print("Training for target " + str(i + 1))

        X_train, X_test, Y_train, Y_test, x_real_test = preprocess(i + 1)

        keras_pred = KerasRegression(X_train, Y_train, X_test, Y_test, x_real_test, i)

        lasso_pred = LassoRegression(X_train, Y_train, X_test, Y_test, x_real_test, i)

        xgb_pred = xgBoostTrain(X_train, Y_train, X_test, Y_test, x_real_test, i)

        file_name = 'hw07_target' + str(i + 1) + '_test_predictions.csv'

