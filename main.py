
import re
from typing import Callable, Tuple
import seaborn as sns
from sklearn import metrics
from sklearn.base import BaseEstimator
import plotly.express as px
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier



def data_preprocessing(A, isTrain):
    """

    :param A: Data frame
    :param isTrain: boolean, True if A is the train data, False otherwise
    :return: the data frame after preprocessing
    """

    males = A.loc[(A["Sex"] == "male")]
    females = A.loc[(A["Sex"] == "female")]
    A = A.drop_duplicates()
    A = A.drop('Ticket', axis=1)
    A = titles_preprocessing(A)
    A['Name'][A["Sex"] == 'male'] = A['Name'][A["Sex"] == 'male'].replace('Miss', None)
    A['Name'][A["Sex"] == 'male'] = A['Name'][A["Sex"] == 'male'].replace(
        'Mrs', None)
    A['Name'][A["Sex"] == 'female'] = A['Name'][A["Sex"] == 'female'].replace(
        'Master', None)
    A['Name'][A["Sex"] == 'female'] = A['Name'][A["Sex"] == 'female'].replace(
        'Mr', None)

    A['Traveling_alone'] = (A['SibSp']+A['Parch'])==0
    A['Traveling_alone']=(A['Traveling_alone']).replace(True, 1)
    A['Traveling_alone'] = (A['Traveling_alone']).replace(False, 0)


    average_age_f = females['Age'].mean()
    average_age_m = males['Age'].mean()
    A['Age'][A["Sex"]=='male'] = (A['Age'][A["Sex"]=='male']).replace(np.nan, average_age_m)
    A['Age'][A["Sex"] == 'female'] = (A['Age'][A["Sex"]=='female']).replace(np.nan, average_age_f)
    A["Age"][A["Age"]<0][A['Sex']=='male'] = average_age_m
    A["Age"][A["Age"] > 100][A['Sex']=='male'] = average_age_m
    A["Age"][A["Age"] < 0][A['Sex'] == 'female'] = average_age_f
    A["Age"][A["Age"] > 100][A['Sex'] == 'female'] = average_age_f
    A = A[A["Pclass"] >0]
    A = A[A["Pclass"] <4]


    average_pclass_f = int(females['Pclass'].mean())
    average_pclass_m = int(males['Pclass'].mean())

    A['Pclass'][A["Sex"] == 'male'] = (A['Pclass'][A["Sex"] == 'male']).replace(
        np.nan, average_pclass_m)
    A['Pclass'][A["Sex"] == 'female'] = (A['Pclass'][A["Sex"] == 'female']).replace(
        np.nan, average_pclass_f)
    #complete cabin according to Pclass to first class, second
    #class and third class
    A = cabin_preprocessing(A)
    for i in range(len(A)):
        if A['Cabin'][i] is np.nan:
            if A['Pclass'][i] == 1:
                A['Cabin'][i] = 'cab_1class'
            if A['Pclass'][i] == 2:
                A['Cabin'][i] = 'cab_2class'
            if A['Pclass'][i] == 3:
                A['Cabin'][i] = 'cab_3class'

    A['Embarked'] = ((A['Embarked'])).replace(np.nan, "unknown")
    A = embarked_preprocessing(A)
    average_SibSp_m = int(males['SibSp'].mean())
    average_SibSp_f = int(females['SibSp'].mean())
    average_Parch_m = int(males['Parch'].mean())
    average_Parch_f = int(females['Parch'].mean())
    A['SibSp'][A["Sex"] == 'female'] = ((A['SibSp'][A["Sex"] == 'female'])).replace(np.nan, average_SibSp_f)
    A['SibSp'][A["Sex"] == 'male'] = (
    (A['SibSp'][A["Sex"] == 'male'])).replace(np.nan, average_SibSp_m)
    A['Parch'][A["Sex"] == 'male'] = ((A['Parch'][A["Sex"] == 'male'])).replace(np.nan, average_Parch_m)
    A['Parch'][A["Sex"] == 'female'] = ((A['Parch'][A["Sex"] == 'female'])).replace(np.nan, average_Parch_f)
    average_Fare_m = int(males['Fare'].mean())
    average_Fare_f = int(females['Fare'].mean())
    A['Fare'][A["Sex"] == 'female'] = ((A['Fare'][A["Sex"] == 'female'])).replace(np.nan, average_Fare_f)
    A['Fare'][A["Sex"] == 'male'] = (
    (A['Fare'][A["Sex"] == 'male'])).replace(np.nan, average_Fare_m)


    A = A.replace({np.nan: None})
    if isTrain:
        A = A.dropna(axis=0)
    A = pd.get_dummies(A, prefix='Cabin', columns=['Cabin'])
    A = pd.get_dummies(A, prefix='Embarked', columns=['Embarked'])
    A = pd.get_dummies(A, prefix='Sex', columns=['Sex'])
    A = pd.get_dummies(A, prefix='Name', columns=['Name'])


    return A


def titles_preprocessing(X):
    """

    :param X: pd data frame
    :return: the data frame after replacing all names by just titles
    """
    X_name = X['Name'].to_numpy()
    for i,name in enumerate(X_name):
        if re.search('Master.',name):
            X_name[i]='Master'
        elif re.search('Miss.', name):
            X_name[i] = 'Miss'
        elif re.search('Mr.', name):
            X_name[i] = 'Mr'
        elif re.search('Mrs.', name):
            X_name[i] = 'Mrs'
        else:
            X_name[i] = 'otherTitle'

    X_name = pd.DataFrame(X_name)
    X['Name'] = X_name
    return X
def embarked_preprocessing(X):
    """
        :param X: pd data frame
        :return: the data frame after replacing all invalid values of embarked
        column with None
        """
    X_embarked = X['Embarked'].to_numpy()
    for i, name in enumerate(X_embarked):
        if X_embarked[i] != 'unknown' and X_embarked[i] != 'Q' and  X_embarked[i] != 'S' and X_embarked[i] != 'C':
            X_embarked[i] = None
    X_embarked = pd.DataFrame(X_embarked)
    X['Embarked'] = X_embarked
    return X


def cabin_preprocessing(X):
    """
        :param X: pd data frame
        :return: the data frame after replacing all cabins by deck
        """

    X_cabin = X['Cabin'].to_numpy()
    for i, cabin in enumerate(X_cabin):
        if(cabin is not np.nan):
            if re.search('A', cabin):
                X_cabin[i] = 'A'
            elif re.search('B', cabin):
                X_cabin[i] = 'B'
            elif re.search('C', cabin):
                X_cabin[i] = 'C'
            elif re.search('D', cabin):
                X_cabin[i] = 'D'
            elif re.search('E', cabin):
                X_cabin[i] = 'E'
            elif re.search('F', cabin):
                X_cabin[i] = 'F'
            elif re.search('G', cabin):
                X_cabin[i] = 'G'
            elif re.search('T', cabin):
                X_cabin[i] = None

    X_cabin = pd.DataFrame(X_cabin)
    X['Cabin'] = X_cabin
    return X


def data_splitting(X: pd.DataFrame, y: pd.Series):
    """

    :param X: pd data frame
    :param y: labels
    :return: X_train, X_test, y_train, y_test, splitted data and labels in 2
    """
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3)

    return (X_train, X_test, y_train, y_test)


def ages_preprocess_for_plots(title, X, column):
    """

    :param title: title of the plot
    :param X: pd data frame
    :param column: name of the column
    :return: void. outputs a plot
    """
    average_age_m = X['Age'].mean()
    ages_t = (X)['Age']
    ages_t = (ages_t).replace(np.nan, average_age_m)
    ages = ages_t.astype(int)
    for i in range(0, 11):
        ages = ages.replace(i, '0-10')
    for i in range(11, 19):
        ages = ages.replace(i, '11-18')
    for i in range(19, 25):
        ages = ages.replace(i, '19-24')
    for i in range(25, 31):
        ages = ages.replace(i, '25-30')
    for i in range(31, 41):
        ages = ages.replace(i, '31-40')
    for i in range(41, 51):
        ages = ages.replace(i, '41-50')
    for i in range(51, 61):
        ages = ages.replace(i, '51-60')
    for i in range(61, 71):
        ages = ages.replace(i, '61-70')
    for i in range(71, 81):
        ages = ages.replace(i, '71-80')
    for i in range(81, 91):
        ages = ages.replace(i, '81-90')
    (X)['Age'] = ages
    X[column].value_counts().plot(kind='bar', title=title,
                                  figsize=(16, 9))
    plt.xticks(fontsize=12)
    plt.show()


def data_vizualization(X: pd.DataFrame):
    """

    :param X: data frame, with survival labels
    :return: void
    this method contains code to output plots.
    """

    males = X.loc[(X["Sex"] == "male")]
    females = X.loc[(X["Sex"] == "female")]
    #males and females data
    X['Sex'].value_counts().plot(kind='bar', title='Gender of passengers',
                                     figsize=(16, 9))
    print(len(males[males['Survived']==1])*(1/len(males)))
    print(len(females[females['Survived'] == 1]) * (1 / len(females)))
    males_survival= X.loc[(X["Sex"] == "male")]['Survived']

    females_survival = (females)['Survived']
    males_survival.value_counts().plot(kind='bar', title='males_survival',
                                 figsize=(16, 9))

    females_survival.value_counts().plot(kind='bar', title='females_survival',
                              figsize=(16, 9))

    #males and females & classes

    m_class =males['Pclass']
    f_class = females['Pclass']

    m_class.value_counts().plot(kind='bar', title='males pclass',
                                   figsize=(16, 9))

    f_class.value_counts().plot(kind='bar', title='females pclass',
                                     figsize=(16, 9))
    f_class_surv = females.loc[(females['Survived'] == 1)]['Pclass']
    m_class_surv = males.loc[(males['Survived'] == 1)]['Pclass']
    f_class_surv.value_counts().plot(kind='bar', title='females survived pclass',
                                     figsize=(16, 9))
    m_class_surv.value_counts().plot(kind='bar',
                                     title='males survived pclass',
                                     figsize=(16, 9))

    #female survivors percentage in each class
    print((1 / len(females.loc[(females['Pclass'] == 3)])) * len(
        females[(females['Pclass'] == 3)][females['Survived'] == 1]))
    print((1/len(females.loc[(females['Pclass'] == 2)]))*len(
        females[(females['Pclass'] == 2)][females['Survived']==1]))
    print((1 / len(females.loc[(females['Pclass'] == 1)])) * len(
        females[(females['Pclass'] == 1)][females['Survived'] == 1]))


    #heatmap of survival, class and age

    A = pd.DataFrame()
    A['Survived'] = X['Survived']
    A['Pclass'] = X['Pclass']
    A['Age'] = X['Age']
    corr = A.corr()
    sns.heatmap(corr, annot=True, square=True)
    plt.yticks(rotation=0)



    #siblings data
    siblings = X['SibSp']
    siblings.value_counts().plot(kind='bar',
                                       title='siblings number',
                                       figsize=(16, 9))
    siblings_survival = X.loc[(X['Survived'] == 1)]['SibSp']
    siblings_survival.value_counts().plot(kind='bar',
                                 title='survivors_siblings_number',
                                 figsize=(16, 9))

    #add travel alove column
    X['Traveling_alone'] = (X['SibSp'] + X['Parch']) == 0
    X['Traveling_alone'] = (X['Traveling_alone']).replace(True, 1)
    X['Traveling_alone'] = (X['Traveling_alone']).replace(False, 0)

    #percentage of survival among those who traveled alone and those who didnt
    print((1/len(X.loc[(X['Traveling_alone'] == 0)]))*len(
        X[(X['Traveling_alone'] == 0)][X['Survived']==1]))
    print((1 / len(X.loc[(X['Traveling_alone'] == 1)])) * len(
        X[(X['Traveling_alone'] == 1)][X['Survived'] == 1]))
    X['Traveling_alone'].value_counts().plot(kind='bar',
                                 title='Traveling alone numbers among survivors',
                                 figsize=(16, 9))

    #A heatmap of siblings, perents/children and class
    A = pd.DataFrame()
    A['SibSp'] = X['SibSp']
    A['Parch'] = X['Parch']
    A['Pclass']=X['Pclass']
    corr = A.corr()
    sns.heatmap(corr, annot=True, square=True)
    plt.yticks(rotation=0)



    #cabins data analysis
    cabins = X['Cabin'].unique()
    df_cabins = pd.DataFrame(cabins, columns=['cabin'])
    survival_rate = np.zeros(len(cabins),)
    for i in range(len(cabins)):
        cabin_survival = (X[X['Cabin']==cabins[i]])
        survival_rate[i]= (len(cabin_survival[cabin_survival['Survived']==1]))/len(cabin_survival)
    df_cabins['survival_rate'] = survival_rate
    fig = px.bar(df_cabins, x='cabin', y='survival_rate',
                    title='survival rate in each cabin')
    fig.show()

    cabin_A = X.loc[(X["Cabin"] == "A")]
    cabin_F = X.loc[(X["Cabin"] == "F")]
    cabin_E = X.loc[(X["Cabin"] == "E")]
    ages_preprocess_for_plots('A deck passengers gender', cabin_A, 'Sex')
    ages_preprocess_for_plots('F deck passengers gender', cabin_F, 'Sex')
    ages_preprocess_for_plots('E deck passengers gender', cabin_E, 'Sex')
    plt.show()

def match_data(X1, X2):
    """

    :param X1: pd data frame
    :param X2: pd data frame
    :return: X1 and X2 after matching their columns to be the same
    """
    X1_cols = X1.columns
    X2_cols = X2.columns
    for col in X2_cols:
        if col not in X1_cols:
            X2 = X2.drop(col, axis=1)
    for col in X1_cols:
        if col not in X2_cols:
            X2[col] = np.zeros(len(X2), )
    X2 = X2.reindex(columns=X1_cols)
    return X1, X2



def cross_validate(estimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds

    """
    av_train_score = 0
    av_validation_score = 0

    X_split = np.array_split(X, cv)
    y_split = np.array_split(y, cv)

    for i in range(cv):
        train = np.array([])
        train_res = np.array([])
        for j in range(cv):
            if(i != j):
                if len(train) == 0:
                    train = X_split[j]
                    train_res = y_split[j]
                else:
                    train = np.concatenate((train, X_split[j]), axis = 0)
                    train_res = np.concatenate((train_res, y_split[j]), axis = None)
        validation = X_split[i]
        validation_res = y_split[i]
        estimator.fit(train,train_res)
        y_train_pred = estimator.predict(train)
        av_train_score+= scoring(train_res, y_train_pred)


        y_valid_pred = estimator.predict(validation)
        av_validation_score += scoring(validation_res, y_valid_pred)
    return av_train_score*(1/cv), av_validation_score*(1/cv)

def loss(y_true, y_pred):
    """

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: error of the prediction
    """
    error = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            error += 1
    return error/len(y_true)










if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("train.csv")
    X = data_preprocessing(df, True)
    X = X.drop('PassengerId', axis=1)
    y = pd.DataFrame(X['Survived'], columns=['Survived'])
    (X_train, X_test, y_train, y_test) = data_splitting(X, y)
    X_train = X_train.drop('Survived', axis =1)
    X_test = X_test.drop('Survived', axis =1)

    #cross validation for choosing depth of trees
    depths = np.arange(1,21)
    err_train_l = np.zeros(len(depths)+1,)
    err_validation_l = np.zeros(len(depths)+1, )
    for i, depth in enumerate(depths):
        err_train, err_validation = cross_validate(RandomForestClassifier(max_depth=depth,
                                                                              min_samples_split=20,
                                                                              criterion='log_loss'), X_train.to_numpy(), y_train.to_numpy(),loss, 5)
        err_validation_l[i] = err_validation
        err_train_l[i] = err_train
    fig = go.Figure([
        go.Scatter(
            name='error train',
            x=depths,
            y=err_train_l,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),

        go.Scatter(
            name='error validation',
            x=depths,
            y=err_validation_l,
            line=dict(color='rgb(214, 39, 40)'),
            mode='lines'
        )
    ])
    fig.update_layout(
        xaxis_title='alpha value',
        yaxis_title='loss',
        title='train&validation errors for different depths',
        hovermode="x"
    )
    fig.show()
    model = RandomForestClassifier(max_depth=5, n_estimators=11,
                                   min_samples_split=20, criterion='log_loss')


    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[::,1]


    fpr, tpr, _ = metrics.roc_curve(y_train.to_numpy(), y_pred_proba)

    # create ROC curve

    plt.plot(fpr, tpr)
    auc = metrics.roc_auc_score(y_train.to_numpy(), y_pred_proba)
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


    y_test_pred = model.predict(X_test)
    print(loss(y_test_pred,y_test.to_numpy()))



    #//making predictions for the unlabelad set
    X = X.drop('Survived', axis =1)
    tData_df = pd.read_csv("test.csv")
    tData_df = data_preprocessing(tData_df, False)

    tData_df_w_pid = tData_df.drop('PassengerId', axis=1)
    X, tData_df_w_pid = match_data(X, tData_df_w_pid)


    model2 = RandomForestClassifier(max_depth=5,n_estimators=11,min_samples_split=20, criterion='log_loss')
    model2.fit(X, y)
    predictions = model2.predict(tData_df_w_pid)


    df_pred = pd.DataFrame()
    df_pred['Survived'] = predictions
    df_pred['PassengerId'] = tData_df['PassengerId']
    df_pred.head()
    df_pred.to_csv("predictions.csv", index=False)

















