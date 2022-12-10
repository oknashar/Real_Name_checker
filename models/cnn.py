import tensorflow as tf
from keras import layers
from keras.models import Model
from shared_param import input_dim,input_length,output_dim

class CNNModel(tf.keras.Model):
    def __init__(self):
        super(CNNModel,self).__init__()
        self.embedding1 = layers.Embedding(input_dim=input_dim, 
                                output_dim=output_dim, 
                                input_length=input_length)
        self.conv1 = layers.Conv1D(filters=16, kernel_size=4, activation='relu')
        self.drop1 = layers.Dropout(0.5)
        self.pool1 = layers.MaxPooling1D(pool_size=2)
        self.flat1 = layers.Flatten()
        self.dense1 = layers.Dense(10, activation='relu')
        self.drop2 = layers.Dropout(0.5)
        self.outputs = layers.Dense(1, activation='sigmoid')
        
    def call(self,input_tensor):

        embedding1 = self.embedding1(input_tensor)
        conv1 = self.conv1(embedding1)
        drop1 = self.drop1(conv1)
        pool1 = self.pool1(drop1)
        flat1 = self.flat1(pool1)
        dense1 = self.dense1(flat1)
        drop2 = self.drop2(dense1)
        outputs = self.outputs(dense1)

        
        return outputs
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def CNNmodel_1(input_shape=(input_length,),optimizer='adam'):
    inputs_tensor = layers.Input(shape=input_shape)
    out = CNNModel()(inputs_tensor)
    model = Model(inputs=inputs_tensor, outputs=out)
    # compile
    model.compile(optimizer=optimizer, 
                loss='binary_crossentropy', 
                metrics=['accuracy'])# summarize
    # print(model.summary())  
    return model
