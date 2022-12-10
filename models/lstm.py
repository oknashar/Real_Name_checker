import tensorflow as tf
from keras import layers
from keras.models import Model

from shared_param import input_dim,input_length,output_dim

class LSTModel(tf.keras.Model):
    def __init__(self):
        super(LSTModel,self).__init__()
        self.embedding1 = layers.Embedding(input_dim=input_dim, 
                           output_dim=output_dim, 
                           input_length=input_length)
        self.lstm = layers.SimpleRNN(30, dropout=0.2,)
        self.GMP = layers.GlobalMaxPool1D()
        self.dense1 = layers.Dense(128, activation='relu')
        self.drop1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(64, activation='relu')
        self.drop2 = layers.Dropout(0.2)
        self.dense3 = layers.Dense(32, activation='relu')
        self.drop3 = layers.Dropout(0.2)
        self.outputs = layers.Dense(1, activation='sigmoid')
        
    def call(self,input_tensor):

        embedding1 = self.embedding1(input_tensor)
        lstm = self.lstm(embedding1)
        dense1 = self.dense1(lstm)
        drop1 = self.drop1(dense1)
        dense2 = self.dense2(drop1)
        drop2 = self.drop2(dense2)
        dense3 = self.dense3(drop2)
        drop3 = self.drop3(dense3)
        out = self.outputs(drop3)

        
        return out

def LSTModel_1(input_shape=(input_length,),optimizer='adam'):
    inputs_tensor = layers.Input(shape=input_shape)
    out = LSTModel()(inputs_tensor)
    model = Model(inputs=inputs_tensor, outputs=out)
            # compile
    model.compile(optimizer=optimizer, 
                loss='binary_crossentropy', 
                metrics=['accuracy'])# summarize
    # print(model.summary())
    return model
