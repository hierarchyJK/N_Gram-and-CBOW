# -*- coding:utf-8 -*-
"""
@project: untitled
@author: KunJ
@file: bag-of-word_torch.py
@ide: Pycharm
@time: 2019-06-21 12:28:27
@month: Jun
"""
import torch
from torch import nn, optim
import torch.nn.functional as F

CONTEXT_SIZE = 2
NUM_DIM = 100
num_epoches = 100
Learning_rate = 1e-3
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)

word_to_idx = {word:i for i, word in enumerate(vocab)}
idx_to_word = {word_to_idx[word]:word for word in word_to_idx}
data = []
for i in range(CONTEXT_SIZE, len(raw_text) -  CONTEXT_SIZE):
    context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

# Start Training
class CBOW(nn.Module):
    def __init__(self, n_word, n_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)
        self.project = nn.Linear(n_dim, n_dim, bias=False)
        self.linear1 = nn.Linear(n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, x):
        x = self.embedding(x)
        x = self.project(x)
        x = torch.sum(x, 0, keepdim=True)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x)
        return x
model = CBOW(len(vocab),NUM_DIM)
print(model)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=Learning_rate)
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 100)
    running_loss = 0.0
    for word in data:
        context, target = word
        context = torch.LongTensor([word_to_idx[i] for i in context])
        target = torch.LongTensor([word_to_idx[target]])
        if use_gpu:
            context = context.cuda()
            target = target.cuda()

        # forward
        out = model(context)
        loss = criterion(out, target)

        running_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss:{:.6f}'.format(running_loss/len(data)))

pred_word_list = []
for word in data:
    context,target = word
    context = torch.LongTensor([word_to_idx[i] for i in context])
    if use_gpu:
        context = context.cuda()
    out = model(context)
    _, pred = torch.max(out, 1)
    predict_word = idx_to_word[pred.item()]
    pred_word_list.append(predict_word)
    print('real word is {}, predict word is {}'.format(target, predict_word))





