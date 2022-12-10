from sklearn.metrics import classification_report
import re
from sklearn.model_selection import train_test_split
from models import model
from helpers import preprocessing, tokenization
from shared_param import input_length,input_dim,model_path

import tensorflow as tf
X,y = preprocessing('Data/df_generated.csv')
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,stratify=y)

X_train,X_test = tokenization(input_dim,X_train,X_test,input_length)
X_train,X_val,y_train,y_val =train_test_split(X_train,y_train,test_size=0.1,stratify=y_train)


model = model()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=True,
                    validation_data=(X_val,y_val),
                    batch_size=512,
                    callbacks=[model_checkpoint_callback])


model.save(model_path)
pred = model.predict(X_test)
pred = pred>0.35
print(classification_report(y_test,pred))