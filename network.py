# some important imports
import numpy as np
from translator import Translator
from tqdm import tqdm

# data
text = open('data/text.txt', 'r').read()
#text = 'Hallo'
text_length = len(text)
characters = list(set(text))

# initializing translator and creating training data
tl = Translator(characters)
X = tl.to_one_hot(text)

# hyperparameter
network_length = X.shape[0]
hidden_size = 100
learning_rate = 1e-1
iterations = 1000

# initializing learnable parameter
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
grad_squared_xh, grad_squared_hh, grad_squared_hy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
for ite in tqdm(range(iterations)):
    y = np.append(X[1:X.shape[0]],X[0])
    loss, dWhh, dWxh, dWhy, ht = forward_and_backward(X, y, ht[-1])
    # adagrad
    grad_squared_xh += dWxh ** 2
    grad_squared_hh += dWhh ** 2
    grad_squared_hy += dWhy ** 2
    Wxh -= dWxh / np.sqrt(grad_squared_xh + 1e-7) * learning_rate
    Whh -= dWhh / np.sqrt(grad_squared_hh + 1e-7) * learning_rate
    Why -= dWhy / np.sqrt(grad_squared_hy + 1e-7) * learning_rate
    if ite % 100 == 0:
        print('Iteration ', ite, ' and loss ', loss)
        print('Sample at iteration')
        print(predict(X, Wxh, Whh, Why, ht[-1]))

