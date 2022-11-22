import tensorflow as tf
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras import optimizers
from keras.layers import GaussianNoise


from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor



def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def r2_keras_loss(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return SS_res/(SS_tot + K.epsilon())

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 


def create_RF_model():
    params = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_split": 5,
    "criterion": "absolute_error",
    'max_features': 10,
    }
    reg_ensemble = ensemble.RandomForestRegressor(**params)

    return reg_ensemble

def create_XGB_model():
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "min_samples_split": 5,
        "learning_rate": 0.1,
        "loss": 'huber', #"squared_error",
        "validation_fraction": 0.2
    }
    reg_ensemble = ensemble.GradientBoostingRegressor(**params)

    return reg_ensemble

def create_ANN_model():
    reg_nn = MLPRegressor(hidden_layer_sizes=(50,30,20,10),
                          activation='relu',
                          solver='lbfgs', 
                          batch_size=100, 
                          max_iter=200,
                          learning_rate='adaptive', 
                          shuffle=True, 
                          validation_fraction=0.1)
    return reg_nn


def create_loso_model(n_features, final):
    model = Sequential()
    
    # Input layer
    model.add(Dense(n_features, input_shape=(n_features,), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    
    if(final):
        # Hidden layers
        model.add(Dense(60, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
    #        
        model.add(Dense(50, kernel_initializer='he_normal'))
        model.add(BatchNormalization())  
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(40, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(30, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
    else:
           # Hidden layers
        model.add(Dense(40, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.2))
    #        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization())  
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1)) 
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.05))
        
        model.add(Dense(5, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.01))
    
    # Output layer
    model.add(Dense(1))
    
    ##### Optimizers  #######
    optimizer = optimizers.RMSprop(lr=0.005)
#    optimizer = optimizers.rmsprop(lr=0.0005)
#    optimizer = optimizers.RMSprop(lr=0.001)
    
    # Compilation
    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[r2_keras])
    
    return model

def create_loyo_model(n_features):
    model = Sequential()
    
    # Input layer
    model.add(Dense(n_features, input_shape=(n_features,), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    
    # Hidden layers
    model.add(Dense(40, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.3))
    
    model.add(Dense(20, kernel_initializer='he_normal'))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.2))
        
    model.add(Dense(10, kernel_initializer='he_normal'))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.1))
    
    model.add(Dense(5, kernel_initializer='he_normal'))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.1))
    
    # Output layer
    model.add(Dense(1))
    
    ##### Optimizers  #######
#    optimizer = optimizers.rmsprop(lr=0.05)
    optimizer = optimizers.RMSprop(lr=0.02)
    
    # Compilation
    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[r2_keras])
#    model.compile(optimizer = optimizer, loss=r2_keras_loss, metrics=[r2_keras])
#    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[r2_keras])

    
    return model

def create_loyso_model(n_features, final):
    model = Sequential()
        
    # Input layer
    model.add(Dense(n_features, input_shape=(n_features,), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    
    # Hidden layers
    if(final):
        # Hidden layers
        model.add(Dense(80, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
    #        
        model.add(Dense(60, kernel_initializer='he_normal'))
        model.add(BatchNormalization())  
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(50, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(30, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(5, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
    else:
        model.add(GaussianNoise(0.1))
        model.add(Dense(40, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.30))
    #    model.add(Dropout(0.1))
        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.20))
    #    model.add(Dropout(0.1))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1))
    #    model.add(Dropout(0.05))
    
        model.add(Dense(5, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1))
    #    model.add(Dropout(0.05))
    
        model.add(Dense(5, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1))
    #    model.add(Dropout(0.05))
    
    # Output layer
    model.add(Dense(1))
        
    ##### Optimizers  #######
    optimizer = optimizers.RMSprop(lr=0.01)
#    optimizer = optimizers.rmsprop(lr=0.002)
#    optimizer = optimizers.rmsprop(lr=0.001)
#    optimizer = optimizers.rmsprop(lr=0.0005)
#    optimizer = optimizers.adam(lr=0.0005)
#    optimizer = optimizers.adam(lr=0.0003)
#    optimizer = optimizers.rmsprop(lr=0.0007)
#    optimizer = optimizers.rmsprop(lr=0.0008)
    
    # Compilation
    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[r2_keras])
#    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[root_mean_squared_error])
#    model.compile(optimizer = optimizer, loss=r2_keras_loss, metrics=[r2_keras])

    
    return model
