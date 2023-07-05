# The model architecture:
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

import keras.backend
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Layer, LSTM, Bidirectional, Dense, Dropout,TimeDistributed, Embedding,Input, Flatten, RepeatVector, ZeroPadding1D, ZeroPadding2D
import tensorflow as tf 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
### DEFINE F2V LAYER ###

class F2V(Layer):
    
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(F2V, self).__init__(**kwargs)
        
    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                                shape=(input_shape[1],6),
                                initializer='uniform',
                                trainable=True)

        self.P = self.add_weight(name='P',
                                shape=(input_shape[1],6),
                                initializer='uniform',
                                trainable=True)

        self.w = self.add_weight(name='w',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        self.p = self.add_weight(name='p',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        super(F2V, self).build(input_shape)
        
    def call(self, x):
        
        original = self.w * x + self.p
        cos_trans = K.cos(K.dot(x, self.W) + self.P)
        
        return K.concatenate([cos_trans, original], -1)
    
class OHANA():
    
    def __init__(self, dim=1,output=6,emb_size=6, pre_train = False, model_weights=None, history=None,**kwargs):
        
        self.dim = dim
        self.output = output
        self.emb_size = emb_size
        
        self.n_Hour =24+1
        self.n_Minute =60+1
        self.n_Day_Of_Month =31+1
        self.n_Day_Of_Week =7+1
        self.n_IW_Wind_Direction_Mark =11+1
        self.n_IW_Weather_Cloudiness_Class =12+1
        self.n_IW_Sunrise_Hour =24+1
        self.n_IW_Sunrise_Minute =60+1
        self.n_IW_Sunset_Hour =24+1
        self.n_IW_Sunset_Minute =60+1
        self.n_TT_P1_Road_Type_Class =4+1
        self.n_TT_P2_Road_Type_Class =4+1
        self.n_TT_P1_Is_Road_Closed =2+1
        self.n_TT_P2_Is_Road_Closed =2+1
        
        if pre_train:
            self.model = self.model_generation()
            self.model.load_weights(model_weights)
        else:
            self.model = self.model_generation()

        if history:
            self.data = pd.read_csv(history)
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
            scaled = self.data.copy()
            scaler = MinMaxScaler(feature_range=(-1, 1)) 
            cont = ['IW_Temp_C','IW_Air_Pressure_mmHg','IW_Air_Humidity_Percent','IW_Wind_Speed_MH','IW_Rain_MMH','TT_P1_Current_Speed_Kmh','TT_P1_Freeflow_Speed_Kmh','TT_P1_Speed_Diff_Kmh','TT_P1_Current_Travel_Time_Sec', 'TT_P1_Freeflow_Travel_Time_Sec','TT_P1_Travel_Time_Diff_Sec']
            for i in cont:
                scaled[i] = scaler.fit_transform(scaled[i].values.reshape(-1, 1))
                self.data = scaled
        

        super(OHANA, self).__init__(**kwargs)
        
    # convert an array of values into a dataset matrix
    def create_dataset(self,dataset):
        dataX, dataY = [], []
        i = 0
        while i < len(dataset)-60-1 :
            a = dataset.iloc[i, :]
            dataX.append(a)
            i = i + 1
            b = [dataset.iloc[i, 1],dataset.iloc[i+4, 1],dataset.iloc[i+9, 1],dataset.iloc[i+14, 1],dataset.iloc[i+29, 1],dataset.iloc[i+59, 1]]
            dataY.append(b)
        return np.array(dataX), np.array(dataY)
    

    def model_generation(self):
    
        ########################### Categorical Embeddings ###########################
        # If we’re in a hurry, one rule of thumb is to use the fourth root of the total number of unique categorical 
        # elements while another is that the embedding dimension should be approximately 1 times the square root
        # of the number of unique elements in the category, and no less than 600.
        
        
        Hour_Input = Input(shape=(1,),name='Hour_Input',dtype='int64')
        E_Hour = Embedding(self.n_Hour,int(1.6*math.sqrt(self.n_Hour)),name='Hour_embedding',embeddings_initializer="he_uniform")(Hour_Input)

        Minute_Input = Input(shape=(1,),name='Minute_Input',dtype='int64')
        E_Minute = Embedding(self.n_Minute,int(1.6*math.sqrt(self.n_Minute)),name='Minute_embedding',embeddings_initializer="he_uniform")(Minute_Input)

        Day_Of_Month_Input = Input(shape=(1,),name='Day_Of_Month_Input',dtype='int64')
        E_Day_Of_Month = Embedding(self.n_Day_Of_Month,int(1.6*math.sqrt(self.n_Day_Of_Month)),name='Day_Of_Month_embedding',embeddings_initializer="he_uniform")(Day_Of_Month_Input)


        Day_Of_Week_Input = Input(shape=(1,),name='Day_Of_Week_Input',dtype='int64')
        E_Day_Of_Week = Embedding(self.n_Day_Of_Week,int(1.6*math.sqrt(self.n_Day_Of_Week)),name='Day_Of_Week_embedding',embeddings_initializer="he_uniform")(Day_Of_Week_Input)


        IW_Wind_Direction_Mark_Input = Input(shape=(1,),name='IW_Wind_Direction_Mark_Input',dtype='int64')
        E_IW_Wind_Direction_Mark = Embedding(self.n_IW_Wind_Direction_Mark,int(1.6*math.sqrt(self.n_IW_Wind_Direction_Mark)),name='IW_Wind_Direction_Mark_embedding',embeddings_initializer="he_uniform")(IW_Wind_Direction_Mark_Input)

        IW_Weather_Cloudiness_Class_Input = Input(shape=(1,),name='IW_Weather_Cloudiness_Class_Input',dtype='int64')
        E_IW_Weather_Cloudiness_Class = Embedding(self.n_IW_Weather_Cloudiness_Class,int(1.6*math.sqrt(self.n_IW_Weather_Cloudiness_Class)),name='IW_Weather_Cloudiness_Class_embedding',embeddings_initializer="he_uniform")(IW_Weather_Cloudiness_Class_Input)


        IW_Sunrise_Hour_Input = Input(shape=(1,),name='IW_Sunrise_Hour_Input',dtype='int64')
        E_IW_Sunrise_Hour = Embedding(self.n_IW_Sunrise_Hour,int(1.6*math.sqrt(self.n_IW_Sunrise_Hour)),name='IW_Sunrise_Hour_embedding',embeddings_initializer="he_uniform")(IW_Sunrise_Hour_Input)

        IW_Sunrise_Minute_Input = Input(shape=(1,),name='IW_Sunrise_Minute_Input',dtype='int64')
        E_IW_Sunrise_Minute = Embedding(self.n_IW_Sunrise_Minute,int(1.6*math.sqrt(self.n_IW_Sunrise_Minute)),name='IW_Sunrise_Minute_embedding',embeddings_initializer="he_uniform")(IW_Sunrise_Minute_Input)


        IW_Sunset_Hour_Input = Input(shape=(1,),name='IW_Sunset_Hour_Input',dtype='int64')
        E_IW_Sunset_Hour = Embedding(self.n_IW_Sunset_Hour,int(1.6*math.sqrt(self.n_IW_Sunset_Hour)),name='IW_Sunset_Hour_embedding',embeddings_initializer="he_uniform")(IW_Sunset_Hour_Input)


        IW_Sunset_Minute_Input = Input(shape=(1,),name='IW_Sunset_Minute',dtype='int64')
        E_IW_Sunset_Minute = Embedding(self.n_IW_Sunset_Minute,int(1.6*math.sqrt(self.n_IW_Sunset_Minute)),name='IW_Sunset_Minute_embedding',embeddings_initializer="he_uniform")(IW_Sunset_Minute_Input)

        TT_P1_Road_Type_Class_Input = Input(shape=(1,),name='TT_P1_Road_Type_Class',dtype='int64')
        E_TT_P1_Road_Type_Class = Embedding(self.n_TT_P1_Road_Type_Class,int(1.6*math.sqrt(self.n_TT_P1_Road_Type_Class)),name='TT_P1_Road_Type_Class_embedding',embeddings_initializer="he_uniform")(TT_P1_Road_Type_Class_Input)


        TT_P2_Road_Type_Class_Input = Input(shape=(1,),name='TT_P2_Road_Type_Class',dtype='int64')
        E_TT_P2_Road_Type_Class = Embedding(self.n_TT_P2_Road_Type_Class,int(1.6*math.sqrt(self.n_TT_P2_Road_Type_Class)),name='TT_P2_Road_Type_Class_embedding',embeddings_initializer="he_uniform")(TT_P2_Road_Type_Class_Input)

        TT_P1_Is_Road_Closed_Input = Input(shape=(1,),name='TT_P1_Is_Road_Closed',dtype='int64')
        E_TT_P1_Is_Road_Closed = Embedding(self.n_TT_P1_Is_Road_Closed,int(1.6*math.sqrt(self.n_TT_P1_Is_Road_Closed)),name='TT_P1_Is_Road_Closed_embedding',embeddings_initializer="he_uniform")(TT_P1_Is_Road_Closed_Input)

        TT_P2_Is_Road_Closed_Input = Input(shape=(1,),name='TT_P2_Is_Road_Closed',dtype='int64')
        E_TT_P2_Is_Road_Closed = Embedding(self.n_TT_P2_Is_Road_Closed,int(1.6*math.sqrt(self.n_TT_P2_Is_Road_Closed)),name='TT_P2_Is_Road_Closed_embedding',embeddings_initializer="he_uniform")(TT_P2_Is_Road_Closed_Input)


        E_Cat_Concat = tf.keras.layers.Concatenate(axis=-1,name='Cat_concat')([E_Hour,E_Minute,E_Day_Of_Month,E_Day_Of_Week,E_IW_Wind_Direction_Mark,E_IW_Weather_Cloudiness_Class,E_IW_Sunrise_Hour,E_IW_Sunrise_Minute,E_IW_Sunset_Hour, E_IW_Sunset_Minute , E_TT_P1_Road_Type_Class , E_TT_P1_Is_Road_Closed , E_TT_P2_Is_Road_Closed ,E_TT_P2_Road_Type_Class])

        ########################### Contineous Embeddings ###########################

        TN_Noise_Level_DB_Input = Input(shape=(1,self.dim),name='TN_Noise_Level_DB_inp')
        TN_Noise_Level_DB_Vec = F2V(self.dim)(TN_Noise_Level_DB_Input)

        IW_Temp_C_Input =  Input(shape=(1,self.dim),name='IW_Temp_C_Input')
        IW_Temp_C_Vec = F2V(self.dim)(IW_Temp_C_Input)

        IW_Air_Pressure_mmHg_Input =  Input(shape=(1,self.dim),name='IW_Air_Pressure_mmHg_Input')
        IW_Air_Pressure_mmHg_Vec = F2V(self.dim)(IW_Air_Pressure_mmHg_Input)

        IW_Air_Humidity_Percent_Input =Input(shape=(1,self.dim),name='IW_Air_Humidity_Percent_Input')
        IW_Air_Humidity_Percent_Vec = F2V(self.dim)(IW_Air_Humidity_Percent_Input)

        IW_Wind_Speed_MH_Input = Input(shape=(1,self.dim),name='IW_Wind_Speed_MH_Input')
        IW_Wind_Speed_MH_Vec = F2V(self.dim)(IW_Wind_Speed_MH_Input)

        IW_Rain_MMH_Input =  Input(shape=(1,self.dim),name='IW_Rain_MMH_Input')
        IW_Rain_MMH_Vec = F2V(self.dim)(IW_Rain_MMH_Input)

        TT_P1_Current_Speed_Kmh_Input = Input(shape=(1,self.dim),name='TT_P1_Current_Speed_Kmh_Input')
        TT_P1_Current_Speed_Kmh_Vec = F2V(self.dim)(TT_P1_Current_Speed_Kmh_Input)

        TT_P1_Freeflow_Speed_Kmh_Input =  Input(shape=(1,self.dim),name='TT_P1_Freeflow_Speed_Kmh_Input')
        TT_P1_Freeflow_Speed_Kmh_Vec = F2V(self.dim)(TT_P1_Freeflow_Speed_Kmh_Input)

        TT_P1_Speed_Diff_Kmh_Input = Input(shape=(1,self.dim),name='TT_P1_Speed_Diff_Kmh_Input')
        TT_P1_Speed_Diff_Kmh_Vec = F2V(self.dim)(TT_P1_Speed_Diff_Kmh_Input)

        TT_P1_Current_Travel_Time_Sec_Input =  Input(shape=(1,self.dim),name='TT_P1_Current_Travel_Time_Sec_Input')
        TT_P1_Current_Travel_Time_Sec_Vec = F2V(self.dim)(TT_P1_Current_Travel_Time_Sec_Input)

        TT_P1_Freeflow_Travel_Time_Sec_Input = Input(shape=(1,self.dim),name='TT_P1_Freeflow_Travel_Time_Sec_Input')
        TT_P1_Freeflow_Travel_Time_Sec_Vec = F2V(self.dim)(TT_P1_Freeflow_Travel_Time_Sec_Input)

        TT_P1_Travel_Time_Diff_Sec_Input =  Input(shape=(1,self.dim),name='TT_P1_Travel_Time_Diff_Sec_Input')
        TT_P1_Travel_Time_Diff_Sec_Vec = F2V(self.dim)(TT_P1_Travel_Time_Diff_Sec_Input)
        
        

        ########################### Vehicle Count KF ###########################

        
        Car_Input =  Input(shape=(1,self.dim),name='Car_Input')
        Car_Vec = F2V(self.dim)(Car_Input)

        Motorbike_Input =  Input(shape=(1,self.dim),name='Motorbike_Input')
        Motorbike_Vec = F2V(self.dim)(Motorbike_Input)

        Bus_Input =  Input(shape=(1,self.dim),name='Bus_Input')
        Bus_Vec = F2V(self.dim)(Bus_Input)

        Truck_Input =  Input(shape=(1,self.dim),name='Truck_Input')
        Truck_Vec = F2V(self.dim)(Truck_Input)
    
        E_Cont_Concat = tf.keras.layers.Concatenate(axis=1, name='Cont_concat')([TN_Noise_Level_DB_Vec,IW_Temp_C_Vec,IW_Air_Pressure_mmHg_Vec,IW_Air_Humidity_Percent_Vec,IW_Wind_Speed_MH_Vec,IW_Rain_MMH_Vec,TT_P1_Current_Speed_Kmh_Vec,TT_P1_Freeflow_Speed_Kmh_Vec,TT_P1_Speed_Diff_Kmh_Vec,TT_P1_Current_Travel_Time_Sec_Vec, TT_P1_Freeflow_Travel_Time_Sec_Vec,TT_P1_Travel_Time_Diff_Sec_Vec,Car_Vec,Motorbike_Vec,Bus_Vec,Truck_Vec])
#         E_Cont_Concat = tf.keras.layers.Concatenate(axis=1, name='Cont_concat')([TN_Noise_Level_DB_Vec,IW_Temp_C_Vec,IW_Air_Pressure_mmHg_Vec,IW_Air_Humidity_Percent_Vec,IW_Wind_Speed_MH_Vec,IW_Rain_MMH_Vec,TT_P1_Current_Speed_Kmh_Vec,TT_P1_Freeflow_Speed_Kmh_Vec,TT_P1_Speed_Diff_Kmh_Vec,TT_P1_Current_Travel_Time_Sec_Vec, TT_P1_Freeflow_Travel_Time_Sec_Vec,TT_P1_Travel_Time_Diff_Sec_Vec])

        ########################### Concatenation: Embeddings+KF ###########################

        E_Cat_Concat = ZeroPadding1D(padding = (0,15))(E_Cat_Concat)



    
        E_Cat_Concat = ZeroPadding1D(padding = (0,11))(E_Cat_Concat)
        E_Cont_Concat = tf.expand_dims(E_Cont_Concat[:, 0, :], axis=2)
        E_Cat_Concat = tf.expand_dims(E_Cat_Concat[:, 0, :], axis=2)
        E_Cont_Concat = ZeroPadding1D(padding = (0,86))(E_Cont_Concat)
        
        
        attention_probs = Dense(93, activation='tanh', name='attention_probs')(E_Cont_Concat)
        concat = tf.keras.layers.Multiply(name='concat')([ E_Cat_Concat, attention_probs])
  

        ########################### Model architecture ###########################

    
        x = Bidirectional(LSTM(100,activation='elu'))(concat)
        x = Dense(256, activation='elu')(x)
        x = Dense(256, activation='elu')(x)
        x = Dense(256, activation='elu')(x)
        x = Dense(256, activation='elu')(x)
        x = Dense(256, activation='elu')(x)
        x = Dense(128, activation='elu')(x)
        x = Dense(128, activation='elu')(x)
        x = Dense(self.output, activation='linear', name='output_layer')(x)
        
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

        model = Model(inputs = [TN_Noise_Level_DB_Input,IW_Temp_C_Input,IW_Air_Pressure_mmHg_Input,IW_Air_Humidity_Percent_Input,IW_Wind_Speed_MH_Input,IW_Rain_MMH_Input,TT_P1_Current_Speed_Kmh_Input,TT_P1_Freeflow_Speed_Kmh_Input,TT_P1_Speed_Diff_Kmh_Input,TT_P1_Current_Travel_Time_Sec_Input, TT_P1_Freeflow_Travel_Time_Sec_Input,TT_P1_Travel_Time_Diff_Sec_Input, Hour_Input,Minute_Input,Day_Of_Month_Input,Day_Of_Week_Input,IW_Wind_Direction_Mark_Input,IW_Weather_Cloudiness_Class_Input,IW_Sunrise_Hour_Input,IW_Sunrise_Minute_Input,IW_Sunset_Hour_Input, IW_Sunset_Minute_Input , TT_P1_Road_Type_Class_Input , TT_P2_Road_Type_Class_Input, TT_P1_Is_Road_Closed_Input , TT_P2_Is_Road_Closed_Input,Car_Input,Motorbike_Input,Bus_Input,Truck_Input], outputs=x)
#         model = Model(inputs = [TN_Noise_Level_DB_Input,IW_Temp_C_Input,IW_Air_Pressure_mmHg_Input,IW_Air_Humidity_Percent_Input,IW_Wind_Speed_MH_Input,IW_Rain_MMH_Input,TT_P1_Current_Speed_Kmh_Input,TT_P1_Freeflow_Speed_Kmh_Input,TT_P1_Speed_Diff_Kmh_Input,TT_P1_Current_Travel_Time_Sec_Input, TT_P1_Freeflow_Travel_Time_Sec_Input,TT_P1_Travel_Time_Diff_Sec_Input, Hour_Input,Minute_Input,Day_Of_Month_Input,Day_Of_Week_Input,IW_Wind_Direction_Mark_Input,IW_Weather_Cloudiness_Class_Input,IW_Sunrise_Hour_Input,IW_Sunrise_Minute_Input,IW_Sunset_Hour_Input, IW_Sunset_Minute_Input , TT_P1_Road_Type_Class_Input , TT_P2_Road_Type_Class_Input, TT_P1_Is_Road_Closed_Input , TT_P2_Is_Road_Closed_Input,Count_Input], outputs=x)
        model.compile(loss='mse', optimizer='Adam', metrics=['mape','mae',rmse,'mse'])
        
        return model   
    
    
    def train(self,batch_size,epochs,dataset):
    
        X , y = self.create_dataset(dataset)


#         kf = KalmanFilter(initial_state_mean=1, n_dim_obs=4)
#         count = kf.em(X[:,27:].astype(np.float32)).smooth(X[:,27:].astype(np.float32))

        # split into train and test sets
        train_size = int(len(X) * 0.80)
        train_size = train_size - train_size%6
        test_size = len(X) - train_size

        self.X_test = X[train_size:train_size+test_size,:]
        self.y_test = y[train_size:train_size+test_size]
#         self.count_test = count[0][train_size:train_size+test_size]

        X = X[0:train_size,:]
        y = y[0:train_size]
#         count = count[0][0:train_size]
        
        train_size = int(len(X) * 0.60)
        train_size = train_size - train_size%6
        val_size = len(X) - train_size
        X_val = X[train_size:train_size+val_size,:]
        y_val = y[train_size:train_size+val_size]
#         count_val = count[train_size:train_size+val_size]
        
        X = X[0:train_size,:]
        y = y[0:train_size]
#         count = count[0:train_size]
        
        
#         count = np.reshape(count, (count.shape[0], 1, count.shape[1]))
#         count_val = np.reshape(count_val, (count_val.shape[0], 1, count.shape[1]))

#         print(X.shape,y.shape,count)
        print(self.model.summary())
        
        batch_size=batch_size
        epochs=epochs
        History = self.model.fit([X[:,1].astype(np.float32),X[:,2].astype(np.float32),X[:,3].astype(np.float32),X[:,4].astype(np.float32),X[:,5].astype(np.float32),X[:,6].astype(np.float32),X[:,7].astype(np.float32),X[:,8].astype(np.float32),X[:,9].astype(np.float32),X[:,10].astype(np.float32),X[:,11].astype(np.float32),X[:,12].astype(np.float32),X[:,13].astype(np.float32),X[:,14].astype(np.float32),X[:,15].astype(np.float32),X[:,16].astype(np.float32),X[:,17].astype(np.float32),X[:,18].astype(np.float32),X[:,19].astype(np.float32),X[:,20].astype(np.float32),X[:,21].astype(np.float32),X[:,22].astype(np.float32),X[:,23].astype(np.float32),X[:,24].astype(np.float32),X[:,25].astype(np.float32),X[:,26].astype(np.float32),X[:,27].astype(np.float32),X[:,28].astype(np.float32),X[:,29].astype(np.float32),X[:,30].astype(np.float32)],y.astype(np.float32), batch_size=batch_size,epochs=epochs, verbose = 0,validation_data=([X_val[:,1].astype(np.float32),X_val[:,2].astype(np.float32),X_val[:,3].astype(np.float32),X_val[:,4].astype(np.float32),X_val[:,5].astype(np.float32),X_val[:,6].astype(np.float32),X_val[:,7].astype(np.float32),X_val[:,8].astype(np.float32),X_val[:,9].astype(np.float32),X_val[:,10].astype(np.float32),X_val[:,11].astype(np.float32),X_val[:,12].astype(np.float32),X_val[:,13].astype(np.float32),X_val[:,14].astype(np.float32),X_val[:,15].astype(np.float32),X_val[:,16].astype(np.float32),X_val[:,17].astype(np.float32),X_val[:,18].astype(np.float32),X_val[:,19].astype(np.float32),X_val[:,20].astype(np.float32),X_val[:,21].astype(np.float32),X_val[:,22].astype(np.float32),X_val[:,23].astype(np.float32),X_val[:,24].astype(np.float32),X_val[:,25].astype(np.float32),X_val[:,26].astype(np.float32),X_val[:,27].astype(np.float32),X_val[:,28].astype(np.float32),X_val[:,29].astype(np.float32),X_val[:,30].astype(np.float32)],y_val.astype(np.float32)))      
        fig, ax = plt.subplots( 1, figsize=(10,9))
        loss_train = History.history['loss']
        ep = range(1,epochs+1)
        plt.plot(ep, loss_train, 'g', label='Training loss')
        plt.title('Training loss (mse)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        self.model.save_weights('OHANA.h5')
        
    
    def predict_evaluation(self,X_test,y_test):

        testPredict = self.model.predict([X_test[:,1].astype(np.float32),X_test[:,2].astype(np.float32),X_test[:,3].astype(np.float32),X_test[:,4].astype(np.float32),X_test[:,5].astype(np.float32),X_test[:,6].astype(np.float32),X_test[:,7].astype(np.float32),X_test[:,8].astype(np.float32),X_test[:,9].astype(np.float32),X_test[:,10].astype(np.float32),X_test[:,11].astype(np.float32),X_test[:,12].astype(np.float32),X_test[:,13].astype(np.float32),X_test[:,14].astype(np.float32),X_test[:,15].astype(np.float32),X_test[:,16].astype(np.float32),X_test[:,17].astype(np.float32),X_test[:,18].astype(np.float32),X_test[:,19].astype(np.float32),X_test[:,20].astype(np.float32),X_test[:,21].astype(np.float32),X_test[:,22].astype(np.float32),X_test[:,23].astype(np.float32),X_test[:,24].astype(np.float32),X_test[:,25].astype(np.float32),X_test[:,26].astype(np.float32),X_test[:,27].astype(np.float32),X_test[:,28].astype(np.float32),X_test[:,29].astype(np.float32),X_test[:,30].astype(np.float32)],verbose = 0)
        
        print('################ Overall evaluation ################')
        testScore = math.sqrt(mean_squared_error(y_test, testPredict))
        print('RMSE: %f' % (testScore))

        mae = mean_absolute_error(y_test, testPredict)
        print('MAE: %f' % mae)

        mse = mean_squared_error(y_test, testPredict)
        print('MSE: %f' % mse)

        mape = np.mean(np.abs((y_test - testPredict) / y_test)) * 100
        print('MAPE: %f' % mape)
        print( '\n')

        print('################ 1 min evaluation ################')

        testScore = math.sqrt(mean_squared_error(y_test[:,0], testPredict[:,0]))
        print('RMSE: %f' % (testScore))

        mae = mean_absolute_error(y_test[:,0], testPredict[:,0])
        print('MAE: %f' % mae)

        mse = mean_squared_error(y_test[:,0], testPredict[:,0])
        print('MSE: %f' % mse)

        mape = np.mean(np.abs((y_test[:,0] - testPredict[:,0]) / y_test[:,5])) * 100
        print('MAPE: %f' % mape)
        print( '\n')

        print('################ 5 min evaluation ################')

        testScore = math.sqrt(mean_squared_error(y_test[:,1], testPredict[:,1]))
        print('RMSE: %f' % (testScore))

        mae = mean_absolute_error(y_test[:,1], testPredict[:,1])
        print('MAE: %f' % mae)

        mse = mean_squared_error(y_test[:,1], testPredict[:,1])
        print('MSE: %f' % mse)

        mape = np.mean(np.abs((y_test[:,1] - testPredict[:,1]) / y_test[:,5])) * 100
        print('MAPE: %f' % mape)
        print( '\n')

        print('################ 10 min evaluation ################')
        testScore = math.sqrt(mean_squared_error(y_test[:,2], testPredict[:,2]))
        print('RMSE: %f' % (testScore))

        mae = mean_absolute_error(y_test[:,2], testPredict[:,2])
        print('MAE: %f' % mae)

        mse = mean_squared_error(y_test[:,2], testPredict[:,2])
        print('MSE: %f' % mse)

        mape = np.mean(np.abs((y_test[:,2] - testPredict[:,2]) / y_test[:,5])) * 100
        print('MAPE: %f' % mape)
        print( '\n')


        print('################ 15 min evaluation ################')
        testScore = math.sqrt(mean_squared_error(y_test[:,3], testPredict[:,3]))
        print('RMSE: %f' % (testScore))

        mae = mean_absolute_error(y_test[:,3], testPredict[:,3])
        print('MAE: %f' % mae)

        mse = mean_squared_error(y_test[:,3], testPredict[:,3])
        print('MSE: %f' % mse)

        mape = np.mean(np.abs((y_test[:,3] - testPredict[:,3]) / y_test[:,5])) * 100
        print('MAPE: %f' % mape)
        print( '\n')

        print('################ 30 min evaluation ################')
        testScore = math.sqrt(mean_squared_error(y_test[:,4], testPredict[:,4]))
        print('RMSE: %f' % (testScore))

        mae = mean_absolute_error(y_test[:,4], testPredict[:,4])
        print('MAE: %f' % mae)

        mse = mean_squared_error(y_test[:,4], testPredict[:,4])
        print('MSE: %f' % mse)

        mape = np.mean(np.abs((y_test[:,4] - testPredict[:,4]) / y_test[:,5])) * 100
        print('MAPE: %f' % mape)
        print( '\n')

        print('################ 60 min evaluation ################')

        testScore = math.sqrt(mean_squared_error(y_test[:,5], testPredict[:,5]))
        print('RMSE: %f' % (testScore))

        mae = mean_absolute_error(y_test[:,5], testPredict[:,5])
        print('MAE: %f' % mae)

        mse = mean_squared_error(y_test[:,5], testPredict[:,5])
        print('MSE: %f' % mse)

        mape = np.mean(np.abs((y_test[:,5] - testPredict[:,5]) / y_test[:,5])) * 100
        print('MAPE: %f' % mape)
        print( '\n')
    def predict(self,X_test):

        return self.model.predict([X_test.iloc[:,1].astype(np.float32),X_test.iloc[:,2].astype(np.float32),X_test.iloc[:,3].astype(np.float32),X_test.iloc[:,4].astype(np.float32),X_test.iloc[:,5].astype(np.float32),X_test.iloc[:,6].astype(np.float32),X_test.iloc[:,7].astype(np.float32),X_test.iloc[:,8].astype(np.float32),X_test.iloc[:,9].astype(np.float32),X_test.iloc[:,10].astype(np.float32),X_test.iloc[:,11].astype(np.float32),X_test.iloc[:,12].astype(np.float32),X_test.iloc[:,13].astype(np.float32),X_test.iloc[:,14].astype(np.float32),X_test.iloc[:,15].astype(np.float32),X_test.iloc[:,16].astype(np.float32),X_test.iloc[:,17].astype(np.float32),X_test.iloc[:,18].astype(np.float32),X_test.iloc[:,19].astype(np.float32),X_test.iloc[:,20].astype(np.float32),X_test.iloc[:,21].astype(np.float32),X_test.iloc[:,22].astype(np.float32),X_test.iloc[:,23].astype(np.float32),X_test.iloc[:,24].astype(np.float32),X_test.iloc[:,25].astype(np.float32),X_test.iloc[:,26].astype(np.float32),X_test.iloc[:,27].astype(np.float32),X_test.iloc[:,28].astype(np.float32),X_test.iloc[:,29].astype(np.float32),X_test.iloc[:,30].astype(np.float32)],verbose = 0)





