from utils.attention import AttentionLayer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Input, Dense,Embedding, Concatenate, TimeDistributed
from tensorflow.keras.models import Model,load_model, model_from_json
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
import pickle as pkl
import numpy as np
from config import SOURCE_LANGUAGE,TARGET_LANGUAGE
import sys

# loading the model architecture and asigning the weights
json_file = open('models/nmt_model-final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_loaded = model_from_json(loaded_model_json, custom_objects={'AttentionLayer': AttentionLayer})
# load weights into new model
model_loaded.load_weights("models/nmt_model_weight-final.h5")

# Load pickle data

with open('processed_data/nmt_data.pkl','rb') as f:
  X_train, y_train, X_test, y_test,max_length_source,max_length_target = pkl.load(f)
  
with open('processed_data/nmt_source_tokenizer.pkl','rb') as f:
  vocab_size_source, source_word2index, sourceTokenizer = pkl.load(f)

with open('processed_data/nmt_target_tokenizer.pkl', 'rb') as f:
  vocab_size_target, targetword2index, targetTokenizer = pkl.load(f)


source_word2index = sourceTokenizer.index_word
targetword2index = targetTokenizer.index_word

latent_dim=500
# encoder inference
encoder_inputs = model_loaded.input[0]  #loading encoder_inputs
# print(model_loaded.input[0])
# sys.exit()
encoder_outputs, state_h, state_c = model_loaded.layers[6].output #loading encoder_outputs

encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])
# print(encoder_model.predict(np.array([[56, 66]])))
# sys.exit()
# decoder inference
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,),name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,),name='input_4')
decoder_hidden_state_input = Input(shape=(32,latent_dim))

# decoder_state_input_h = Input(shape=(latent_dim,))
# decoder_state_input_c = Input(shape=(latent_dim,))
# decoder_hidden_state_input = Input(shape=(32,latent_dim))

# Get the embeddings of the decoder sequence
decoder_inputs = model_loaded.layers[3].output

dec_emb_layer = model_loaded.layers[5]

dec_emb2= dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_lstm = model_loaded.layers[7]
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_layer = model_loaded.layers[8]
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])

concate = model_loaded.layers[9]
decoder_inf_concat = concate([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_dense = model_loaded.layers[10]
decoder_outputs2 = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
[decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
[decoder_outputs2] + [state_h2, state_c2])


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # return 'here'

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = targetword2index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
          break
        else:
          sampled_token = targetword2index[sampled_token_index]

          if(sampled_token!='end'):
              decoded_sentence += ' '+sampled_token

              # Exit condition: either hit max length or find stop word.
              if (sampled_token == 'end' or len(decoded_sentence.split()) >= (26-1)):
                  stop_condition = True

          # Update the target sequence (of length 1).
          target_seq = np.zeros((1,1))
          target_seq[0, 0] = sampled_token_index

          # Update internal states
          e_h, e_c = h, c

    return decoded_sentence


def test_unseen_data(sentence):
    test_source_seq = sourceTokenizer.texts_to_sequences([sentence])
    return test_source_seq
    prediction = decode_sequence(test_source_seq)

    return prediction


# test = input(f'Enter {SOURCE_LANGUAGE} Sentence: ')
test = 'Where are you';
print(f'Predicted {TARGET_LANGUAGE} Sentence: ',test_unseen_data(test))