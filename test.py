import argparse
import pickle
import sys
import time
import pandas as pd
from helpers import clean_text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from shared_param import tokenizer_path,model_path,input_length
from models import model



def predict_dl(model,tokenizer,text):
    text =clean_text(text)
    text =pd.Series(text)
    text1=tokenizer.texts_to_sequences(text)
    text1=pad_sequences(text1, padding='post', maxlen=input_length)
    p=model.predict(text1)[0][0]
    return p

def model_and_tokenizer(model_path,tokenizer_path,model_structure):
    model_dl = model_structure()
    model_dl.load_weights(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer =pickle.load(f)
    return model_dl , tokenizer

def test_model(model_path , text,model_structure):
    model_dl = model_structure()
    model_dl.load_weights(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer =pickle.load(f)

    return predict_dl(model_dl,tokenizer,text)


# print(test_model(model_path,"عمر خالد النشار",model))
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="the Arabic name ")
    args = vars(ap.parse_args())

    name = args["name"]
    normalized_name = list(name)

    if name:

        start_time = time.time()
        # text 
        text = "".join(name)
        #make model prediction 
        prediction_result=test_model(model_path,text,model)
        fulltime =round(time.time() - start_time,2)


        if prediction_result > 0.35:


            final= {
                "الاسم":text,
                "الحاله":"الاسم صحيح",
                "واثق بنسبة":str(prediction_result),
                "الوقت المحتسب لتنفيذ العمليه":str(fulltime)
            }
            
            print(final)
        else:

            final= {
                "الاسم":text,
                "الحاله":"الاسم غير صحيح",
                "واثق بنسبة":str(1-prediction_result),
                "الوقت المحتسب لتنفيذ العمليه":str(fulltime)
            }
            
            print(final)