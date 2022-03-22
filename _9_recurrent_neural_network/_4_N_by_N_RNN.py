import torch
from torch import nn
from torch import optim


def sample():
    """
    다대다 RNN 예제
    """

    words = "Repeat is the best medicine for memory".split()
    vocab = list(set(words))

    # 각 단어에 정수 부여
    wtoi = {tkn: i for i, tkn in enumerate(vocab, 1)}
    wtoi['<unk>'] = 0
    itow = {v: k for k, v in wtoi.items()}

    # 단어 리스트를 정수 리스트로 변환
    encoded = [wtoi[token] for token in words]

    # 정수 리스트를 X 와 Y 로 분리
    input_seq, label_seq = encoded[:-1], encoded[1:]
    # Repeat is the best medicine for
    X = torch.LongTensor(input_seq).unsqueeze(0)
    # is the best medicine for memory
    Y = torch.LongTensor(label_seq).unsqueeze(0)

    # 하이퍼파라미터 세팅
    vocab_size = len(wtoi)
    # 임베딩 차원 크기
    input_size = 5
    hidden_size = 20

    model = Net(vocab_size, input_size, hidden_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters())

    for epoch in range(201):

        output = model(X)
        loss = loss_function(output, Y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("[{:02d}/201] {:.4f} ".format(epoch + 1, loss))
        pred = output.softmax(-1).argmax(-1).tolist()
        decoded = [itow[p] for p in pred]
        print(" ".join(["Repeat"] + decoded))
        print()


class Net(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        self.rnn_layer = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # 임베딩 층
        # (배치 크기, 시퀀스 길이) -> (배치 크기, 시퀀스 길이, 임베딩 차원)
        output = self.embedding_layer(x)

        # RNN 층
        # (배치 크기, 시퀀스 길이, 임베딩 차원) ->
        # output: (배치 크기, 시퀀스 길이, 은닉층 크기)
        # hidden: (1, 배치 크기, 은닉층 크기)
        output, hidden = self.rnn_layer(output)

        # 최종 출력 층
        # (배치 크기, 시퀀스 길이, 은닉층 크기) => (배치 크기, 시퀀스 길이, 단어장 크기)
        output = self.linear(output)

        # (배치 크기, 시퀀스 길이, 단어장 크기) -> (배치 크기 * 시퀀스 길이, 단어장 크기)
        return output.view(-1, output.size(2))
