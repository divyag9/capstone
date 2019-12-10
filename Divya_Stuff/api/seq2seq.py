import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from attention import AttentionLayer
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session

DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 10

latent_dim = 300
embedding_dim=100

class Seq2SeqSummarizer(object):

    model_name = 'seq2seq'

    def __init__(self, config):
        self.source_tokenizer = config['source_tokenizer']
        self.target_tokenizer = config['target_tokenizer']
        self.reverse_target_word_index = config['reverse_target_word_index']
        self.reverse_source_word_index = config['reverse_source_word_index']
        self.target_word_index = config['target_word_index']
        self.source_word_index = config['source_word_index']
        self.max_text_len = config['max_text_len']
        self.max_summary_len = config['max_summary_len']
        self.reviews_dataframe = config['reviews_dataframe']
        self.config = config
        
#         config_proto = tf.ConfigProto()
#         off = rewriter_config_pb2.RewriterConfig.OFF
#         config_proto.graph_options.rewrite_options.arithmetic_optimization = off
#         config_proto.graph_options.rewrite_options.memory_optimization = off
        
#         session = tf.Session(config=config_proto)
#         K.set_session(session)

        # Encoder
        encoder_inputs = Input(shape=(self.max_text_len,))
        embeddings_index = self.get_glove_embeddings()
        embedding_matrix_x = self.get_embedding_matrix(embeddings_index, self.source_word_index)

        enc_emb = Embedding(len(self.source_word_index)+1,
                            embedding_dim,
                            weights=[embedding_matrix_x],
                            input_length=self.max_text_len,
                            trainable=False)(encoder_inputs)
        #LSTM 1 
        encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True) 
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb) 

        #LSTM 2 
        encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True) 
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1) 

        #LSTM 3 
        encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True) 
        encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        embedding_matrix_y = self.get_embedding_matrix(embeddings_index, self.target_word_index)
        dec_emb_layer = Embedding(len(self.target_word_index)+1,
                            embedding_dim,
                            weights=[embedding_matrix_y],
                            input_length=self.max_summary_len,
                            trainable=False)
        dec_emb = dec_emb_layer(decoder_inputs)

        decoder_lstm = LSTM(latent_dim, 
                            return_sequences=True, 
                            return_state=True,
                            dropout=0.4,
                            recurrent_dropout=0.2,
                            name='decoder_lstm_layer')
        decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=encoder_states)
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
        decoder_dense =  TimeDistributed(Dense(len(self.target_word_index)+1, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_concat_input)

        # Define the model 
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        self.model = model
        
        
        self.encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_hidden_state_input = Input(shape=(self.max_text_len,latent_dim))

        # Get the embeddings of the decoder sequence
        dec_emb_layer = Embedding(len(self.target_word_index)+1,
                                    embedding_dim,
                                    weights=[embedding_matrix_y],
                                    input_length=self.max_summary_len,
                                    trainable=False,
                                    name='decoder_emb_layer')
        dec_emb2= dec_emb_layer(decoder_inputs) 
        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, 
                                                            initial_state=[decoder_state_input_h, decoder_state_input_c])

        #attention inference

        attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = decoder_dense(decoder_inf_concat) 

        # Final decoder model
        self.decoder_model = Model(
            [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs2] + [state_h2, state_c2])

    def get_glove_embeddings(self):
        embeddings_index = {}
        with open('glove.6B.100d.txt', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        return embeddings_index
    
    def get_embedding_matrix(self, embeddings_index, tokenizer_word_index):
        embedding_matrix = np.zeros((len(tokenizer_word_index)+1, embedding_dim))
        for word, i in tokenizer_word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                
        return embedding_matrix
    
    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)
            self.graph = tf.get_default_graph()

    def transform_source_text(self, source_texts):
        text_seq = self.source_tokenizer.texts_to_sequences(source_texts)
        transformed_text = pad_sequences(text_seq, maxlen=self.max_text_len, padding='post')
        return transformed_text

    def transform_target_text(self, target_texts):
        text_seq = self.target_tokenizer.texts_to_sequences(target_texts)
        transformed_text = pad_sequences(text_seq, maxlen=self.max_summary_len, padding='post')
        return transformed_text

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                decoder_input_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqSummarizer.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqSummarizer.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqSummarizer.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        
#         init = tf.global_variables_initializer()
#         self.session.run(init)

        config_file_path = Seq2SeqSummarizer.get_config_file_path(model_dir_path)
        weight_file_path = Seq2SeqSummarizer.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = Seq2SeqSummarizer.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_text(Ytrain)
        Ytest = self.transform_target_text(Ytest)

        Xtrain = self.transform_source_text(Xtrain)
        Xtest = self.transform_source_text(Xtest)

        history = self.model.fit([Xtrain,Ytrain[:,:-1]], 
                                Ytrain.reshape(Ytrain.shape[0],Ytrain.shape[1], 1)[:,1:],
                                epochs=epochs,
                                callbacks=[checkpoint],
                                batch_size=batch_size, 
                                validation_data=([Xtest,Ytest[:,:-1]],Ytest.reshape(Ytest.shape[0],Ytest.shape[1], 1)[:,1:]))

        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        
        transformed_text = self.transform_source_text([input_text])
        with self.graph.as_default():
            e_out, e_h, e_c = self.encoder_model.predict(transformed_text)


            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1,1))

            # Populate the first word of target sequence with the start word.
            target_seq[0, 0] = self.target_word_index['sostok']

            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:

                output_tokens, h, c = self.decoder_model.predict([target_seq] + [e_out, e_h, e_c])

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_token = self.reverse_target_word_index[sampled_token_index]

                if(sampled_token!='eostok'):
                    decoded_sentence += ' '+sampled_token

                # Exit condition: either hit max length or find stop word.
                if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (self.max_summary_len-1)):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1,1))
                target_seq[0, 0] = sampled_token_index

                # Update internal states
                e_h, e_c = h, c

            return decoded_sentence