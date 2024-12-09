import torch
import torch.nn as nn

class PowerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PowerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # x: (batch_size, seq_len, input_size)
        # lengths: (batch_size)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        # 使用最后一个隐藏状态进行预测
        hn = hn[-1]  # (batch_size, hidden_size)
        out = self.fc(hn)  # (batch_size, output_size)
        return out