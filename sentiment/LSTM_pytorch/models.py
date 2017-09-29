import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cPickle
import random
from multiprocessing import Process
from keras.preprocessing import sequence
import time

class BasicRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, args, vocab_size, pretrained=None):
        super(BasicRNN, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.encoder = nn.Embedding(self.vocab_size, self.args.embedding_size)
        self.rnn = nn.LSTM(self.args.embedding_size, self.args.rnn_size, self.args.rnn_layers, bias=False)
        self.decoder = nn.Linear(self.args.rnn_size, 2)
        self.softmax = nn.Softmax()
        #self.AttentionLayer = AttentionLayer(self.args, self.args.rnn_size)
        start = time.clock()
        self.init_weights(pretrained=pretrained)
        print("Initialized LSTM model")

    def init_weights(self, pretrained):
        initrange = 0.1
        if(pretrained is not None):
            print("Setting Pretrained Embeddings")
            pretrained = pretrained.astype(np.float32)
            pretrained = torch.from_numpy(pretrained)
            if(self.args.cuda):
                pretrained = pretrained.cuda()
            self.encoder.weight.data = pretrained
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        #start = time.clock()
        emb = self.encoder(input)
        #print "emb.size()", emb.size()
        #end = time.clock()
        #print "time: ", end - start
        output, hidden = self.rnn(emb, hidden)
        #print "output.size()", output.size()
        last = Variable(torch.LongTensor([output.size()[0]-1]))
        if(self.args.cuda):
            last = last.cuda()
        if(self.args.aggregation=='mean'):
            output = torch.mean(output, 0)
        elif(self.args.aggregation=='last'):
            output = torch.index_select(output,0,last)
        #print output
        output = output.view(-1, 300)
        decoded = self.decoder(output)
        decoded = self.softmax(decoded)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_()),
                Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_()))
