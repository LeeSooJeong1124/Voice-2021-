
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


class RegClassifier1(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=6):
        super(RegClassifier1, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=3, stride=2)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(755904, 1000)
        self.fc2 = nn.Linear(1000, 4)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.pad(x, pad=(0, 0, 2, 2), mode="constant", value=0)
        x = self.conv1(x)
        x = F.pad(x, pad=(0, 0, 1, 1), mode="constant", value=0)
        x = F.relu(self.pool1(x))
        x = F.pad(x, pad=(0, 1, 2, 2), mode="constant", value=0)
        x = self.conv2(x)
        x = F.pad(x, pad=(0, 0, 1, 1), mode="constant", value=0)
        x = F.relu(self.pool2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

class RegClassifier2(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=6): # 7아니면 8
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, channel_1, (5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(channel_1, channel_2, (3, 3), padding=1, bias=True)
        self.fc1 = nn.Linear(channel_2 * 14 * 400, 1000)  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
        self.fc2 = nn.Linear(1000, num_classes)
        self.flatten = nn.Flatten()

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

        pass

    def forward(self, x):
        scores = None

        hidden_1 = F.relu(self.conv1(x))
        hidden_2 = F.relu(self.conv2(hidden_1))
        hidden_2 = self.flatten(hidden_2)

        out = F.relu(self.fc1(hidden_2))
        scores = self.fc2(out)

        return scores


class RegClassifier3(nn.Module):
    def __init__(self, input_size=40, hidden_size=16, num_layers=2, batch_size=20, num_classes=8):
        super(RegClassifier3, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, x):
        #h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float())
        #c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float())
        out, _ = self.rnn(x, self.h0)
        h_t = out[:, -1, :]
        out = self.fc1(h_t)
        #out = h_t.view(-1, self.num_classes)
        return out

class RNN_D(nn.Module):
    def __init__(self, input_size=42, hidden_size=16, num_layers=2, batch_size=20, num_classes=6):
        super(RNN_D, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.LSTM = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        x = x.view(-1, 400)
        out, _ = self.LSTM(x, (self.h0, self.c0))
        h_t = out[:, -1, :]
        out = self.fc1(h_t)
        return out        
    
# CnnLSTM

class CnnLSTM_D(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8, hidden_size=16, num_layers=2, batch_size=20, num_classes=6):
        super(CnnLSTM_D, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM = nn.LSTM(14, self.hidden_size, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, num_classes)
        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = x.view(self.batch_size, -1, 400).transpose(2, 1)

        out, _ = self.LSTM(x, (self.h0, self.c0))
        h_t = out[:, -1, :]
        out = self.fc1(h_t)

        return out
    
class CnnLSTM_Gender(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8, hidden_size=16, num_layers=2, batch_size=20, num_classes=2):
        super(CnnLSTM_Gender, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)
        
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM = nn.LSTM(7, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(self.hidden_size, num_classes)
        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)

    
class CnnLSTM2(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8, hidden_size=16, num_layers=2, batch_size=20, num_classes=8):
        super(CnnLSTM2, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)
        
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM = nn.LSTM(7, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(self.hidden_size, num_classes)
        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        
        x = self.pool(x)

        x = x.view(self.batch_size, -1, 200).transpose(2, 1)

        out, _ = self.LSTM(x, (self.h0, self.c0))
        h_t = out[:, -1, :]
        out = self.fc1(h_t)

        return out
    
class CnnLSTM_Dialect(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8, hidden_size=16, num_layers=2, batch_size=20, num_classes=6):
        super(CnnLSTM_Dialect, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM = nn.LSTM(14, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(self.hidden_size, num_classes)
        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = x.view(self.batch_size, -1, 400).transpose(2, 1)

        out, _ = self.LSTM(x, (self.h0.detach(), self.c0.detach()))
        h_t = out[:, -1, :]
        out = self.fc1(h_t)

        return out
    

class CnnLSTM3(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, hidden_size=16, num_layers=4, batch_size=20, num_classes=6):
        super(CnnLSTM3, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)
        
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)
        
        self.flatten = nn.Flatten()

        self.LSTM = nn.LSTM(7, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(3200, 1024)
        self.fc2 = nn.Linear(1024, self.num_classes)
        self.fc3 = nn.Linear(200, self.num_classes)
        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        
        x = self.pool(x)
        
        x = x.view(self.batch_size, -1, 200).transpose(2, 1)

        out, _ = self.LSTM(x, (self.h0.detach(), self.c0.detach()))
        
        out = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2)
        out = self.fc3(out)

        return out

    
    
class CNN_D(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=6):
        super(CNN_D, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(channel_1)
        self.bn2 = nn.BatchNorm2d(channel_2)

        # self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9504, 100, bias=True)
        self.fc2 = nn.Linear(100, self.num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class RegClassifierG2(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=2):
        super(RegClassifierG2, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(channel_1)
        self.bn2 = nn.BatchNorm2d(channel_2)

        # self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9504, 100, bias=True)
        self.fc2 = nn.Linear(100, self.num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class RegClassifierD(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=6):
        super(RegClassifierD, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(channel_1)
        self.bn2 = nn.BatchNorm2d(channel_2)

        # self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9504, 100, bias=True)
        self.fc2 = nn.Linear(100, self.num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def check_accuracy(loader, model, dataset='valid'):
    dataset = dataset
    if dataset == 'train':
        print('Checking accuracy on train set')
    elif dataset == 'valid':
        print('Checking accuracy on validation set')
    elif dataset == 'test':
        print('Checking accuracy on test set')
    num_correct = 0
    
    ac = []
    
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for dict in loader:
            x = dict['input']
            y = dict['label']
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        
        ac.append(acc)
        
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        
        return ac


def tester(model, batch_size, model_type="CNN"):
    if model_type == "CNN":
        x = torch.zeros((batch_size, 3, 13, 400), dtype=dtype)  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    elif model_type == "RNN":
        x = torch.zeros((batch_size, 400, 39), dtype=dtype)  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    test_model = model
    scores = test_model(x)
    print(scores.size())


def train(model, loader_train, loader_valid, tb_path, optimizer, epochs=1, print_every=50):
    model = model.to(device=device)
    writer = tb_path

    running_loss = 0.0
    
    acc = []
    acc_val = []
    loss_train = []
        
    epoch_loss = 0.0
    dialect_weight = torch.tensor([0.13, 0.7, 0.2, 1, 1.5, 1.5], dtype=torch.float32)
    dialect_weight = dialect_weight.to(device=device)

    for e in range(epochs):
        model.train()
        for t, dict in enumerate(loader_train):
            
            x = dict['input']
            y = dict['label']
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.int64)

            scores = model(x)
            crit = nn.CrossEntropyLoss(weight=dialect_weight)
            loss = crit(scores, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            writer.add_scalar('training loss', running_loss / 10, epochs * len(loader_train) + t)
            running_loss = 0
            
            #if t % print_every == 0:
            print('Iteration %d --- Train Loss = %.4f' % (t+1, loss.item()))

            #if t % print_every == (t-1):
        print("Epoch %d finished" % (e+1))
        tmp = check_accuracy(loader=loader_train, model=model, dataset='train')
        acc.append(tmp)
        tmp = check_accuracy(loader=loader_valid, model=model, dataset='valid')
        acc_val.append(tmp)
        loss_train.append(loss)
    
    import matplotlib.pyplot as plt
    plt.plot(acc,'-')
    plt.plot(acc_val,'-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.show()

    
    plt.plot(loss_train)
    plt.ylabel('loss_train')
    plt.xlabel('epoch')
    plt.title('Model loss') 
    plt.show()
    
    print('---- FINISHED!! ----')
