# some important imports
import numpy as np
from tqdm import tqdm

# data
text = open('data/text.txt', 'r').read()

# text = 'Hallo'
text_length = len(text)
chars = list(set(text))
char_length = len(chars)
print('text is ', text_length, 'long and has ', char_length)

# creating training data
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

X = np.array([char_to_int[char] for char in text])
y = np.append(X[1:X.shape[0]], X[0])
print('first 10 datas: ', X[0:10])
print('first 10 labels: ', y[0:10])

# initializing hyperparameter
seq_size = 15
hidden_size = 200
learning_rate = 1e-8
epochs = 5000
print('Training ', epochs, ' epochs with a sequence size of ', seq_size, ', a hidden size of ', hidden_size,
      ' and a learning rate of', learning_rate)

# initializing learnable parameter
Wxh = np.random.randn(hidden_size, char_length) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(char_length, hidden_size) * 0.01


def forward_pass(X, hprev):
    ht, pt, loss = [hprev[0]], [], 0
    for t in range(len(X)):
        # creating a one hot encoded vector
        xt = np.zeros((char_length, 1))
        xt[X[t]] = 1

        # calculating forward pass
        zt = np.dot(Wxh, xt) + np.dot(Whh, ht[t])
        ht.append(np.tanh(zt))
        yt = np.dot(Why, ht[t])

        # getting probability distribution
        pt.append(np.exp(yt) / np.sum(np.exp(yt)))

        # summing up the loss of every output
        loss += -np.sum(np.log(pt[t][X[t]]))
    return ht, pt, loss / len(X)


def backward_pass(X, y, ht, pt):
    dWhh, dWxh, dWhy = np.zeros_like(Whh), np.zeros_like(Wxh), np.zeros_like(Why)
    for t in reversed(range(len(X))):
        dout = pt.copy()
        dout[t][y[t]] -= 1
        dWhy += np.dot(dout[t], ht[t].T)
        dh = np.dot(Why.T, dout[t])
        dtanh = (1 - ht[t] * ht[t]) * dh
        xt = np.zeros((char_length, 1))
        xt[X[t]] = 1
        dWxh += np.dot(dtanh, xt.T)
        dWhh += np.dot(dtanh, ht[t - 1].T)

    dWhh /= len(X)
    dWxh /= len(X)
    dWhy /= len(X)
    # gradient clipping
    for dparam in [dWxh, dWhh, dWhy]:
        np.clip(dparam, -5, 5, out=dparam)
    return dWhh, dWxh, dWhy


def predict(X, Wxh, Whh, Why, hprev):
    ht, prediction = [hprev[0]], ''
    for t in range(len(X)):
        # creating a one hot encoded vector
        xt = np.zeros((char_length, 1))
        xt[X[t]] = 1

        # calculating forward pass
        zt = np.dot(Wxh, xt) + np.dot(Whh, ht[t])
        ht.append(np.tanh(zt))
        yt = np.dot(Why, ht[t])

        # getting probability distribution
        pt.append(np.exp(yt) / np.sum(np.exp(yt)))

        # creating a prediction string
        prediction += chars[np.argmax(pt[t])]
    return prediction


# initializing hidden state and squared gradient
ht = [np.zeros((hidden_size, 1))]
grad_squared_xh, grad_squared_hh, grad_squared_hy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)

loss = 0
for e in tqdm(range(epochs)):
    for steps in range(0, len(X), seq_size):
        inputs = X[steps:steps + seq_size]
        targets = y[steps:steps + seq_size]

        # forward and backward pass
        ht, pt, loss = forward_pass(inputs, ht)
        dWhh, dWxh, dWhy = backward_pass(inputs, inputs, ht, pt)

        # adagrad
        grad_squared_xh += dWxh ** 2
        grad_squared_hh += dWhh ** 2
        grad_squared_hy += dWhy ** 2

        # parameter update
        Wxh -= dWxh / np.sqrt(grad_squared_xh + 1e-7) * learning_rate
        Whh -= dWhh / np.sqrt(grad_squared_hh + 1e-7) * learning_rate
        Why -= dWhy / np.sqrt(grad_squared_hy + 1e-7) * learning_rate

    if e % 100 == 0:
        print('loss at epoch ', e, ' is ', loss)
        prediction = predict(X, Wxh, Whh, Why, ht)
        text_file = open(str(e) + '.txt', 'w')
        text_file.write(prediction)
        text_file.close()
