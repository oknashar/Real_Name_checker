import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

def clean_text(text):  
    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى",
            "\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا",
            "","","","ي","",' ', ' ',' ',' ? ',' ؟ ', ' ! ']

    special_chars=['\n','\t','&quot;','?','؟','!','.','،',',','؛']
    #remove tashkeel
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel,"", text)

            
    for i in range(0, len(search)):
            text = text.replace(search[i], replace[i])

            
            
    text = text.strip()

    return text


def preprocessing(file_path='../df_generated.csv'):

    data = pd.read_csv(file_path)

    data['name2']= data['name'].apply(clean_text)

    X= data['name2']
    y = data['status'].astype(int)

    return X, y

