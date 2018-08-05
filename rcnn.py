# some important imports
import numpy as np
from translator import Translator
from tqdm import tqdm

# hyperparameter
network_length = 29 * 2
hidden_size = 200
learning_rate = 1e-5
iterations = 2000

# data
text = open('data/text.txt', 'r').read()
text_length = len(text)
characters = list(set(text))

# initializing translator and creating training data
tl = Translator(characters)
X = tl.to_one_hot(text)

Wxh = np.random.randn(hidden_size, tl.characters_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(tl.characters_size, hidden_size) * 0.01

def forward_and_backward(X, targets, hprev):
    # forward pass
    zt, ht, yt, pt, loss = [], [], [], [], 0
    ht.append(hprev)
    for t in range(X.shape[0]):
        zt.insert(t, np.dot(Wxh, X[t].reshape(len(characters), 1)) + np.dot(Whh, ht[t - 1]))
        ht.insert(t, np.tanh(zt[t]))
        yt.insert(t, np.dot(Why, ht[t]))
        pt.insert(t, np.exp(yt[t] - np.max(yt[t])) / np.sum(np.exp(yt[t] - np.max(yt[t]))))
        loss += -np.sum(np.log(pt[t])* targets[t])

    # backward pass
    dWhh, dWxh, dWhy = np.zeros_like(Whh), np.zeros_like(Wxh), np.zeros_like(Why)
    for t in reversed(range(X.shape[0])):
        dout = np.copy(pt[t])
        dout[targets[t]] -= 1
        dWhy += np.dot(dout, ht[t].T)
        dh = np.dot(Why.T, dout)
        dtanh = (1 - ht[t] * ht[t]) * dh
        dWxh += np.dot(dtanh, X[t].reshape(len(characters), 1).T)
        dWhh += np.dot(dtanh, ht[t - 1].T)
    return loss, dWhh, dWxh, dWhy, ht
    
def predict(X, Wxh, Whh, Why, hprev):
    zt, ht, yt, pt = [], [], [], []
    ht.append(hprev)
    prediction = ''
    for t in range(X.shape[0]):
        zt.insert(t, np.dot(Wxh, X[t].reshape(len(characters), 1)) + np.dot(Whh, ht[t - 1]))
        ht.insert(t, np.tanh(zt[t]))
        yt.insert(t, np.dot(Why, ht[t]))
        pt.insert(t, np.exp(yt[t] - np.max(yt[t])) / np.sum(np.exp(yt[t] - np.max(yt[t]))))
        prediction += characters[np.argmax(pt[t])]
    return prediction
    
print('starting learning process')
ht = [np.zeros((hidden_size, 1))]
loss = 0
for ite in range(iterations):
    if ite % 100 == 0:
        sample = predict(X, Wxh, Whh, Why, ht[-1])
        print('Sample at ' + str(ite) + ' with a loss of ' + str(loss))
        print(sample)
    for i in range(0, X.shape[0], network_length):
        X_train = X[i:i + network_length]
        y_train = X[i + 1:i + 1 + network_length]
        if y_train.shape[0] != network_length:
            y_train = np.append(y_train, X[0])
        loss, dWhh, dWxh, dWhy, ht = forward_and_backward(X_train, y_train, ht[-1])

        Wxh -= dWxh * learning_rate
        Whh -= dWhh * learning_rate
        Why -= dWhy * learning_rate
        i += network_length
