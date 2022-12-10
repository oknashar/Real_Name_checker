import pickle
from tensorflow.keras.models import load_model

def model_and_tokenizer(model_path,tokenizer_path,model_structure):
    model_dl = model_structure()
    model_dl.load_weights(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer =pickle.load(f)
    return model_dl , tokenizer
