from keras import Input, Model
from keras.layers import Embedding, Dense, Dot, Reshape
import numpy as np

"""Build Skipgram model through Keras"""
vocab_size = 46971
vector_dim = 126
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')

target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

dot_product = Dot(axes=1)([target, context])
dot_product = Reshape((1,))(dot_product)

output = Dense(1, activation='sigmoid')(dot_product)
model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

pos = np.load("Train_data_positive.npy")
neg = np.load("Train_data_negative.npy")
pos_tag = np.ones(len(pos))
neg_tag = np.zeros(len(neg))
final_data = np.concatenate((pos,neg))
final_label = np.concatenate((pos_tag,neg_tag))
model.fit([final_data[:,0],final_data[:,1]], final_label, batch_size = 15000)
model.save("Embedding_Model")

