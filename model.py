
import torch
import torch.nn as nn
import numpy as np
import config
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim, #隐藏层节点数：256
            num_layers=1,
            batch_first=True
            # bidirectional – If True, becomes a bidirectional LSTM. Default: False
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim, #256 √
            hidden_size=embedding_dim, #128
            num_layers=1,
            batch_first=True
        )
    def forward(self, x):
        # print(f"forwarding, x.shape={x.shape}") # [512, 200, 1]
        # x = x.reshape((-1, self.seq_len, self.n_features)) # 
        x = x.reshape((-1, self.seq_len, self.n_features))
        # print(f"forwarding, x.shape after reshape={x.shape}") # [512,200,1] -> [512, 100, 2]

        # output, (hidden_state, cell_state)
        x, (_, _) = self.rnn1(x)
        # print(f"x after rnn1: {x.shape}") # [512, 100, 256]
        x, (hidden_n, _) = self.rnn2(x)
        # print(f"x after rnn2: {x.shape}") # [512, 100(200), 128]
        # print(f"hidden shape after rnn2: {hidden_n.shape}") # [1, 512, 128]
        # return hidden_n.reshape((-1, self.n_features, self.embedding_dim)) # [512,1,128] -> [512,2,128]? (256,2,128)
        return hidden_n.reshape((-1, 1, self.embedding_dim))

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim, #128
            hidden_size=input_dim, #128
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim, #128
            hidden_size=self.hidden_dim, #256
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
    def forward(self, x): #[512,1,128]
        # print("decoder ===")
        # print(x.shape)
        # x = x.repeat(1, self.seq_len, self.n_features)
        x = x.repeat(1, self.seq_len, 1)
        # print(f"after repeat:{x.shape}") # [512,100,128]
        # print(x.shape)
        x = x.reshape(-1, self.seq_len, self.input_dim)
        # print(x.shape)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        # print(f"x shape after rnn1 rnn2: {x.shape}") # [512,100,256]
        x = x.reshape((-1, self.seq_len, self.hidden_dim)) # [512,100,256]
        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(config.device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(config.device)
    def forward(self, x):
        x = self.encoder(x)
        # print(f"hidden_n after lstm={x.shape}") # [256,2,128]
        x = self.decoder(x)
        return x

