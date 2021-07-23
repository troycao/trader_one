from tensorflow import keras
import tensorflow as tf
from utils import process_w2v_data
from visual import show_w2v_word_embedding

corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]

class SkipGram(keras.Model):
    def __init__(self, v_dim, emb_dim):
        super().__init__()
        self.v_dim = v_dim
        self.embeddings = keras.layers.Embedding(
            input_dim=v_dim,
            output_dim=emb_dim,
            embeddings_initializer=keras.initializers.RandomNormal(0, 0,1)
        )

        self.nec_w = self.add_weight(
            name="nec_w",
            shape=[v_dim, emb_dim],
            initializer=keras.initializers.TruncatedNormal(0, 0.1)
        )
        self.nec_b = self.add_weight(
            name="nec_b",
            shape=(v_dim,),
            initializer=keras.initializers.Constant(0.1)
        )
        self.opt = keras.optimizers.Adam(0.01)

    def call(self, x, training=None, mask=None):
        return self.embeddings(x)

    def loss(self, x, y, training=None):
        embedded = self.call(x, training)
        return tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.nec_w,
                biases=self.nec_b,
                labels=tf.expand_dims(y, axis=1),
                inputs=embedded,
                num_sampled=5,
                num_classes=self.v_dim
            )
        )

    def step(self, x, y):
        with tf.GradientTape() as tape:
            loss = self.loss(x, y, True)
            grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()

def train(model,data):
    for t in range(5000):
        bx, by = data.sample(8)
        loss = model.step(bx, by)
        if t % 200 ==0:
            print("step:{} | loss:{}".format(t, loss))

if __name__ == '__main__':
    d = process_w2v_data(corpus, skip_window=2, method="skip_gram")
    m = SkipGram(d.num_word, 2)
    train(m, d)

    show_w2v_word_embedding(m, d, "/Users/troy/work/ai/trader_one/com/troy/data/result/skip-gram.png")