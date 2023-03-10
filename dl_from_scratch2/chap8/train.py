import sys
sys.path.append('..')
sys.path.append('../chap7')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from attention_seq2seq import AttentionSeq2seq
from chap7.seq2seq import Seq2Seq
from chap7.peeky_seq2seq import PeekySeq2Seq

(X_train, t_train), (X_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

X_train, X_test = X_train[:, ::-1], X_test[:, ::-1]

vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch) :
    trainer.fit(X_train, t_train, max_epoch = 1, batch_size = batch_size, max_grad = max_grad)

    correct_num = 0
    for i in range(len(X_test)) :
        question, correct = X_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose, is_reverse = True)

    acc = float(correct_num) / len(X_test)
    acc_list.append(acc)

    print('validation accuracy %.3f%%' % (acc * 100))

model.save_params()

x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim(-0.05, 1.05)
plt.show()