import torch
import torch.nn as nn
import torchtext.vocab as vocab
from data import vocab_size

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained=False):
        super().__init__()
        # Khởi tạo embedding layer (dùng GloVe nếu pretrained=True)
        # [Sinh viên bổ sung: dùng nn.Embedding, xử lý pretrained với GloVe]
        if pretrained:
            # Tải GloVe embeddings
            glove = vocab.GloVe(name='6B', dim=embedding_dim)
            # Khởi tạo embedding layer với GloVe weights
            self.embedding = nn.Embedding.from_pretrained(glove.vectors, freeze=False, padding_idx=0)
        else:
            # Nếu không dùng pretrained, khởi tạo embedding layer thông thường
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Khởi tạo khối RNN layer
        # [Sinh viên bổ sung: dùng nn.RNN với batch_first=True]
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)

        # Khởi tạo tầng Dense để dự đoán 3 nhãn
        # [Sinh viên bổ sung: dùng nn.Linear, nhận hidden state từ RNN]
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # Chuyển text thành embedding
        # [Sinh viên bổ sung]
        embedded = self.embedding(text)
        # Đưa qua khối RNN để lấy hidden state cuối
        # [Sinh viên bổ sung]
        ouput, hidden = self.rnn(embedded)
        # Lấy hidden state cuối cùng (từ bước cuối cùng của RNN)
        hidden = hidden.squeeze(0)
        # Đưa hidden state qua tầng Dense để dự đoán 3 nhãn
        # [Sinh viên bổ sung]
        prediction = self.fc(hidden)

        return prediction # [Sinh viên bổ sung: trả về kết quả dự đoán]

# Khởi tạo mô hình
model = RNNModel(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, output_dim=3, pretrained=True)