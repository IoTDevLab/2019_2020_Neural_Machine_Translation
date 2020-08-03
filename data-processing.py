import pandas as pd
# from .config import MAX_SEQ_LENGTH,FILE_NAME,REMOVE_NUMBERS
import re
from string import digits
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle as pkl

MAX_SEQ_LENGTH = 50 # very important to truncate or things will get too large and bog down the system

FILE_NAME = 'en-tw.txt' # Name of the file with data

REMOVE_NUMBERS = True


print('Data Processing Started')
lines= pd.read_table('data/'+FILE_NAME, names=['source', 'target'])


# check dimensions
# print('shape of data:'+lines.shape)


# Remove extra spaces
lines.source=lines.source.apply(lambda x: x.strip())
lines.target=lines.target.apply(lambda x: x.strip())
lines.source=lines.source.apply(lambda x: re.sub(" +", " ", x))
lines.target=lines.target.apply(lambda x: re.sub(" +", " ", x))

# Remove all numbers from text
if REMOVE_NUMBERS:
    
    remove_digits = str.maketrans('', '', digits)
    lines.source=lines.source.apply(lambda x: x.translate(remove_digits))
    lines.target=lines.target.apply(lambda x: x.translate(remove_digits))
    
    
# truncate to MAX_SEQ_LENGTH
lines.source=lines.source.apply(lambda x: " ".join(x.split(' ')[:MAX_SEQ_LENGTH]))
lines.target=lines.target.apply(lambda x: " ".join(x.split(' ')[:MAX_SEQ_LENGTH]))

# Add start and end tokens to target sequences
lines.target = lines.target.apply(lambda x : "start "+ x + " end")

# Vocabulary of Source
all_source_words=set()
for s in lines.source:
    for word in s.split():
        if word not in all_source_words:
            all_source_words.add(word)

# Vocabulary of Target 
all_target_words=set()
for t in lines.target:
    for word in t.split():
        if word not in all_target_words:
            all_target_words.add(word)
            
            
# Max Length of source sequence
lenght_list=[]
for l in lines.source:
    lenght_list.append(len(l.split(' ')))
max_length_src = np.max(lenght_list)

# Max Length of target sequence
lenght_list=[]
for l in lines.target:
    lenght_list.append(len(l.split(' ')))
max_length_tar = np.max(lenght_list)


input_words = sorted(list(all_source_words))
target_words = sorted(list(all_target_words))
num_encoder_tokens = len(all_source_words)
num_decoder_tokens = len(all_target_words)


num_decoder_tokens += 1 # For zero padding
num_decoder_tokens


input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])

reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())

lines = shuffle(lines)

# Train - Test Split
X, y = lines.source, lines.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


def Max_length(data):
    max_length_ = max([len(x.split(' ')) for x in data])
    return max_length_

#Training data
max_length_source = Max_length(X_train)
max_length_target = Max_length(y_train)

#Test data
max_length_source_test = Max_length(X_test)
max_length_target_test = Max_length(y_test)


sourceTokenizer = Tokenizer()
sourceTokenizer.fit_on_texts(X_train)
source_word2index = sourceTokenizer.word_index
vocab_size_source = len(source_word2index) + 1

X_train = sourceTokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_length_source, padding='post')

X_test = sourceTokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen = max_length_source, padding='post')


targetTokenizer = Tokenizer()
targetTokenizer.fit_on_texts(y_train)
targetword2index = targetTokenizer.word_index
vocab_size_target = len(targetword2index) + 1

y_train = targetTokenizer.texts_to_sequences(y_train)
y_train = pad_sequences(y_train, maxlen=max_length_target, padding='post')

y_test = targetTokenizer.texts_to_sequences(y_test)
y_test = pad_sequences(y_test, maxlen = max_length_target, padding='post')


with open('processed_data/nmt_data.pkl','wb') as f:
  pkl.dump([X_train, y_train, X_test, y_test],f)
with open('processed_data/nmt_source_tokenizer.pkl','wb') as f:
  pkl.dump([vocab_size_source, source_word2index, sourceTokenizer], f)

with open('processed_data/nmt_target_tokenizer.pkl', 'wb') as f:
  pkl.dump([vocab_size_target, targetword2index, targetTokenizer], f)


print('Data Processing Finished')