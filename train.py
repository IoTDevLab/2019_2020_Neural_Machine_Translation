from utils.attention import AttentionLayer
from tensorflow.keras.layers import LSTM, Input, Dense,Embedding, Concatenate, TimeDistributed
from tensorflow.keras.models import Model,load_model, model_from_json
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import pickle as pkl
import numpy as np
# from data-processing import 
from keras import backend as K 
K.clear_session() 

EPOCHS = 2

print('Model Training Started')

def Max_length(data):
    max_length_ = max([len(x.split(' ')) for x in data])
    return max_length_

with open('processed_data/nmt_data.pkl','rb') as f:
  X_train, y_train, X_test, y_test = pkl.load(f)

with open('processed_data/nmt_source_tokenizer.pkl','rb') as f:
  vocab_size_source, source_word2index, sourceTokenizer = pkl.load(f)
  
with open('processed_data/nmt_target_tokenizer.pkl', 'rb') as f:
  vocab_size_target, targetword2index, targetTokenizer = pkl.load(f)

#Training data
max_length_source = len(X_train)
max_length_target = len(y_train)

#Test data
max_length_source_test = len(X_test)
max_length_target_test = len(y_test)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train[0], y_train[0]

latent_dim = 500 

# Encoder 
encoder_inputs = Input(shape=(max_length_source,)) 
enc_emb = Embedding(vocab_size_source, latent_dim,trainable=True)(encoder_inputs) 

#LSTM 1 
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True) 
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb) 

#LSTM 2 
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True) 
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1) 

#LSTM 3 
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True) 
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 

# Set up the decoder. 
decoder_inputs = Input(shape=(None,)) 
dec_emb_layer = Embedding(vocab_size_target, latent_dim,trainable=True) 
dec_emb = dec_emb_layer(decoder_inputs) 

#LSTM using encoder_states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) 
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c]) 

#Attention Layer
attn_layer = AttentionLayer(name='attention_layer') 
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs]) 

# Concat attention output and decoder LSTM output 
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Dense layer
decoder_dense = TimeDistributed(Dense(vocab_size_target, activation='softmax')) 
decoder_outputs = decoder_dense(decoder_concat_input) 

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fit modelq
# path = 'models/nmt-atten-model_{epoch:02d}-{val_acc:.2f}.h5'
# checkpoint = ModelCheckpoint(path, monitor='val_acc', verbose=1,save_best_only=True, mode='max')

history = model.fit([X_train, y_train[:,:-1]], y_train.reshape(y_train.shape[0], y_train.shape[1],1)[:,1:], 
                    epochs=EPOCHS, 
                    # callbacks=[checkpoint],
                    batch_size=32,
                    validation_data = ([X_test, y_test[:,:-1]], y_test.reshape(y_test.shape[0], y_test.shape[1], 1)[:,1:])
                    )

model_json = model.to_json()
with open("models/nmt_model-final.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/nmt_model_weight-final.h5")


print("Model finished training and saved model to disk")