import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys
sys.path.append("../utils")
from ReadData import seqfile_to_instances

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

########################################
# DATA
########################################

X_train_seqs_pos = seqfile_to_instances('../data/TIS/seqs/X_train_TISseqs_pos.txt')[:1000]
X_train_seqs_neg = seqfile_to_instances('../data/TIS/seqs/X_train_TISseqs_neg.txt')[:1000]
X_val_seqs_pos = seqfile_to_instances('../data/TIS/seqs/X_val_TISseqs_pos.txt')[:1000]
X_val_seqs_neg = seqfile_to_instances('../data/TIS/seqs/X_val_TISseqs_neg.txt')[:1000]
X_test_seqs_pos = seqfile_to_instances('../data/TIS/seqs/X_test_TISseqs_pos.txt')[:1000]
X_test_seqs_neg = seqfile_to_instances('../data/TIS/seqs/X_test_TISseqs_neg.txt')[:1000]


# merge train data
# label positive data as 1, negative as 0
Y_train_seqs_pos = np.ones(len(X_train_seqs_pos), dtype=int)
Y_train_seqs_neg = np.zeros(len(X_train_seqs_neg), dtype=int)
X_train = np.concatenate([X_train_seqs_pos, X_train_seqs_neg])
y_train = np.concatenate([Y_train_seqs_pos, Y_train_seqs_neg])

# merge val data
# label positive data as 1, negative as 0
Y_val_seqs_pos = np.ones(len(X_val_seqs_pos), dtype=int)
Y_val_seqs_neg = np.zeros(len(X_val_seqs_neg), dtype=int)
X_val = np.concatenate([X_val_seqs_pos,X_val_seqs_neg])
y_val = np.concatenate([Y_val_seqs_pos,Y_val_seqs_neg])

# merge test data
# label positive data as 1, negative as 0
Y_test_seqs_pos = np.ones(len(X_test_seqs_pos), dtype=int)
Y_test_seqs_neg = np.zeros(len(X_test_seqs_neg), dtype=int)
X_test = np.concatenate([X_test_seqs_pos,X_test_seqs_neg])
y_test = np.concatenate([Y_test_seqs_pos,Y_test_seqs_neg])





##############################################
# MODEL
##############################################

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(203,))
embedding_layer = TokenAndPositionEmbedding(203, 4, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
print(X_train[0])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train, y_train, batch_size=32, epochs=2, validation_data=(X_val, y_val)
)