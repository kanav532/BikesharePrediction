import argparse
import os
import sys
import warnings
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

TRAIN_SPLIT = 8000
SEED = 101
past_history = 24
future_target = 0
BATCH_SIZE = 256
BUFFER_SIZE = 8000
STEPS_PER_EPOCH = 200
EPOCHS = 10
def temperature_segments(value):
    if value < 5:
        return 2.5
    elif value < 15  and value > 5:
        return 10
    elif value < 25 and value > 15:
        return 20
    elif value < 35 and value > 25:
        return 30
    else:
        return 40

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    
    start_index = start_index + history_size
    
    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step = False):
    data = []
    labels = []
    
    start_index = start_index + history_size
    
    if end_index is None:
        end_index = len(dataset) - target_size
     

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i: i + target_size])
    return np.array(data), np.array(labels)

def create_time_steps(length):
    return (list(range(-length, 0)))

def mwa(history):
    return np.mean(history)
def data_processing(filepath):
    df= pd.read_csv(filepath)
    df['day']= df['datetime'].apply(lambda x: x[8:10])
    df['month']= df['datetime'].apply(lambda x: int(x[5:7]))
    df['hour']= df['datetime'].apply(lambda x: int(x[11:13]))
    temp_max = df['temp'].max()
    temp_min = df['temp'].min()
    df['Temperature_converted'] = df['temp'].apply(lambda x: x * (temp_max - temp_min) + temp_min )
    df['Temperature_segments'] = df['temp'].apply(temperature_segments)
    df['day'] = df['day'].apply(lambda x: int(x))
    df.drop(['casual', 'registered'], axis = 1, inplace = True)
    df['dteday'] = df['datetime'].apply(lambda x: x[:10])

    return df 

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'bx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    
    if delta: 
        future = delta
    else:
        future = 0
        
    plt.figure(figsize = (20, 10))    
    plt.title(title, fontsize = 20)

    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize = 10, label = labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label = labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])

def plot_function(dataframe):
    fig, axes = plt.subplots(1, 2, figsize = (15, 5))
    sns.lineplot(ax = axes[0], x = dataframe.columns[0], y = 'Mean Absolute Error', data = dataframe)
    axes[0].set_title('Number of Components Vs Mean Absolute Error', fontsize = 15)
    sns.lineplot(ax = axes[1], x = dataframe.columns[0], y = 'Mean Squared Error', data = dataframe)
    axes[1].set_title('Number of Components Vs Mean Squared Error', fontsize = 15)
    fig.tight_layout()

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))
    plt.figure(figsize = (20, 10))
    plt.plot(epochs, loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.title(title, fontsize = 15)
    plt.legend()
    plt.grid()
    
    plt.show()


def train_univariate(df,):
    univariate_data_df = df['count']
    univariate_data_df.index = df['dteday']
    uni_data = univariate_data_df.values

    tf.random.set_seed(SEED)
    uni_train_mean = uni_data[: TRAIN_SPLIT].mean()
    uni_train_std = uni_data[: TRAIN_SPLIT].std()
    uni_data = (uni_data - uni_train_mean) / uni_train_std
    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT, past_history, future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None, past_history, future_target)
    show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
    i = 20
    show_plot([x_train_uni[i], y_train_uni[i], mwa(x_train_uni[i])], 0, 'Moving Average Prediction')
    i = 25
    show_plot([x_train_uni[i], y_train_uni[i], mwa(x_train_uni[i])], 0, 'Moving Average Prediction')


    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    simple_lstm_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(32, input_shape = x_train_uni.shape[-2:]),
                                              tf.keras.layers.Dense(1)])

    simple_lstm_model.compile(optimizer = 'adam', loss = 'mae')


    simple_lstm_model.fit(train_univariate, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH, validation_data = val_univariate, 
                        validation_steps = 50)
    for x, y in val_univariate.take(5):
        plot = show_plot([x[0].numpy(), y[0].numpy(), simple_lstm_model.predict(x)[0]], 0, 
                        'Simple LSTM model')
                                                    



def train_multivariate(df,):
    features_considered = ['season', 'holiday', 'workingday', 'windspeed', 'temp']
    features = df[features_considered]
    features.index = df['dteday']
    features.head()
    dataset = features.values
    data_mean = dataset[: TRAIN_SPLIT].mean(axis = 0)
    data_std = dataset[: TRAIN_SPLIT].std(axis = 0)

    dataset = (dataset - data_mean) / data_std

    STEP = 1

    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0, TRAIN_SPLIT, past_history,
                                                    future_target, STEP, single_step = True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1], TRAIN_SPLIT, None, past_history,
                                                future_target, STEP, single_step = True)
    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
    single_step_model = tf.keras.models.Sequential()

    single_step_model.add(tf.keras.layers.LSTM(60, 
                                            input_shape = x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))
    single_step_model.compile(optimizer = tf.keras.optimizers.RMSprop(), loss = 'mae')
    single_step_history = single_step_model.fit(train_data_single, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH,
                                           validation_data = val_data_single,
                                           validation_steps = 50)
    plot_train_history(single_step_history, 'Training Mean Absolute Error')

def train_ML_data(df):
    df.drop(['dteday','datetime'], axis = 1, inplace = True)
    X = df.drop(['count'], axis = 1).values
    y = df['count'].values
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.3, random_state = 101)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_cv = scaler.transform(X_cv)
    return X_train, X_cv, y_train, y_cv

def train_DNN(X_train, X_cv, y_train, y_cv):
    model = Sequential()
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(25, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(5, activation = 'relu'))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'MSE', metrics = ['MSE', 'MAE'])
    model.fit(X_train, y_train, epochs = EPOCHS, verbose = 1, validation_data = (X_cv, y_cv))
    plt.figure(figsize = (10, 10))
    sns.regplot(y_cv, model.predict(X_cv), marker = '+', color = 'maroon')
    plt.title('Y_test Vs Y_predictions', color = 'maroon', fontsize = 15)
    plt.savefig('train.png')

def train_KNN(X_train, X_cv, y_train, y_cv):
    model = KNeighborsRegressor()
    mean_squared_error_list = []
    mean_absolute_error_list = []
    roc_auc_score_list = []
    K_nearest_neighbors = [2, 3, 5, 8, 10, 11, 15, 20]
    for i in K_nearest_neighbors:
        model = KNeighborsRegressor(n_neighbors = i)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_cv)
        mean_squared_error_list.append(mean_squared_error(y_predict, y_cv))
        mean_absolute_error_list.append(mean_absolute_error(y_predict, y_cv))
    knn_dictionary = {'K Nearest Neighbors': K_nearest_neighbors, 'Mean Squared Error': mean_squared_error_list, 'Mean Absolute Error': mean_absolute_error_list}
    knn_dataframe = pd.DataFrame(knn_dictionary)
    print(knn_dataframe)
    best_neighbor_index = np.argmin(mean_squared_error_list)
    best_neighbor = K_nearest_neighbors[best_neighbor_index]
    model = KNeighborsRegressor(n_neighbors = best_neighbor)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_cv)
    plt.figure(figsize = (10, 10))
    sns.regplot(y_cv, y_predict, color = 'brown')
    plt.title('Y_test Vs Y_predictions', fontsize = 15, color = 'brown')
    plt.savefig('train.png')

def train_PLSReg(X_train, X_cv, y_train, y_cv):
    n_components_list = [2, 3, 5, 7, 10]
    mean_squared_error_list = []
    mean_absolute_error_list = []
    for i in n_components_list:
        model = PLSRegression(n_components = i)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_cv)
        mean_squared_error_list.append(mean_squared_error(y_predict, y_cv))
        mean_absolute_error_list.append(mean_absolute_error(y_predict, y_cv))
    pls_regression_dict = {'Number of Components': n_components_list, 'Mean Absolute Error': mean_absolute_error_list,
                        'Mean Squared Error': mean_squared_error_list}
    pls_regression_dataframe = pd.DataFrame(pls_regression_dict)
    print(pls_regression_dataframe)
    best_n_components_index = np.argmin(mean_squared_error_list)
    best_n_components = n_components_list[best_n_components_index]
    model = PLSRegression(n_components = best_n_components)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_cv)
    plt.figure(figsize = (10, 10))
    sns.regplot(y_cv, y_predict, color = 'brown')
    plt.title('Y_test Vs Y_predictions', fontsize = 15, color = 'brown')
    plt.savefig('train.png')

def train_DecisionTreeReg(X_train, X_cv, y_train, y_cv):
    max_depth_list = [10, 15, 16, 17, 18, 20, 25]
    mean_absolute_error_list = []
    mean_squared_error_list = []
    for i in max_depth_list:
        model = DecisionTreeRegressor(max_depth = i)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_cv)
        mean_absolute_error_list.append(mean_absolute_error(y_predict, y_cv))
        mean_squared_error_list.append(mean_squared_error(y_predict, y_cv))
    decision_tree_dict = {'Max Depth': max_depth_list, 'Mean Absolute Error': mean_absolute_error_list,
                        'Mean Squared Error': mean_squared_error_list}
    decision_tree_dataframe = pd.DataFrame(decision_tree_dict)
    print(decision_tree_dataframe)
    best_max_depth_index = np.argmin(mean_absolute_error_list)
    best_max_depth = max_depth_list[best_max_depth_index]
    model = DecisionTreeRegressor(max_depth = best_max_depth)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_cv)
    plt.figure(figsize = (10, 10))
    sns.regplot(y_predict, y_cv, marker = '*', color = 'green')
    plt.title('Y_test Vs Y_predictions', fontsize = 15, color = 'green')
    plt.savefig('train.png')

def train_GradientBoostingReg(X_train, X_cv, y_train, y_cv):
    n_estimators_list = [25, 50, 100, 150, 200, 400, 1000]
    mean_squared_error_list = []
    mean_absolute_error_list = []
    for i in n_estimators_list:
        model = GradientBoostingRegressor(n_estimators = i, max_depth = 10)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_cv)
        mean_squared_error_list.append(mean_squared_error(y_cv, y_predict))
        mean_absolute_error_list.append(mean_absolute_error(y_cv, y_predict))
    gradient_boosting_regressor_dict = {"Number of Estimators": n_estimators_list, "Mean Absolute Error": mean_absolute_error_list,
                                    "Mean Squared Error": mean_squared_error_list}
    gradient_boosting_regressor_dataframe = pd.DataFrame(gradient_boosting_regressor_dict) 
    print(gradient_boosting_regressor_dataframe)  
    n_estimators_index = np.argmin(mean_squared_error_list)
    best_n_estimators = n_estimators_list[n_estimators_index]
    model = GradientBoostingRegressor(n_estimators = best_n_estimators, max_depth = 10)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_cv)
    plt.figure(figsize = (10, 10))
    sns.regplot(y_predict, y_cv, marker = 'o', color = 'blue')
    plt.title("Y_test Vs Y_predictions", fontsize = 15, color = 'blue')
    plt.savefig('train.png')

def train_LogReg(X_train, X_cv, y_train, y_cv):
    mean_squared_error_list = []
    mean_absolute_error_list = []
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_cv)
    mean_squared_error_list.append(mean_squared_error(y_predict, y_cv))
    mean_absolute_error_list.append(mean_absolute_error(y_predict, y_cv))
    print(f"The mean squared error of the linear regression model is {mean_squared_error_list[0]}")
    print(f"The mean absolute error of the linear regression model is {mean_absolute_error_list[0]}")
    plt.figure(figsize = (10, 10))
    sns.regplot(y_cv, y_predict, color = 'orange')
    plt.title("Y_test Vs Y_predictions", fontsize = 15, color = 'orange')
    plt.savefig('train.png')


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', required=True)
    parser.add_argument('--algorithm', type=str, required=True,help="Choose the algorithm type amongst dnn,logreg,gbreg,dtreg,plsreg,knn")
    # parser.add_argument('--x', type=int, default=,)
    # parser.add_argument('--y', type=int, default=)  
    args = parser.parse_args()
    data_file= args.filepath
    algorithm= args.algorithm
    df= data_processing(data_file)
    X_train, X_cv, y_train, y_cv= train_ML_data(df)
    if algorithm=="dnn":
        train_DNN(X_train, X_cv, y_train, y_cv)
    if algorithm=="logreg":
        train_LogReg(X_train, X_cv, y_train, y_cv)
    if algorithm=="gbreg":
        train_GradientBoostingReg(X_train, X_cv, y_train, y_cv)
    if algorithm=="dtreg":
        train_DecisionTreeReg(X_train, X_cv, y_train, y_cv)
    if algorithm=="pslreg":
        train_PLSReg(X_train, X_cv, y_train, y_cv)
    if algorithm=="knn":
        train_KNN(X_train, X_cv, y_train, y_cv)