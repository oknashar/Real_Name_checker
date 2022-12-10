
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
from shared_param import tokenizer_path


def tokenization (number_of_words,X_train,X_test,pad_max_len):
    # tokenizer = Tokenizer(num_words=number_of_words)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    # saving
    pickle.dump(tokenizer, open(tokenizer_path,'wb'))
    print("vocab size:",len(tokenizer.word_index))

    X_train = pad_sequences(X_train, padding='post', maxlen=pad_max_len)
    X_test = pad_sequences(X_test, padding='post', maxlen=pad_max_len)

    return X_train ,X_test
