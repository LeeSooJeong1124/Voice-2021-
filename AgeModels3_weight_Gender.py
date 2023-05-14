
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import time
import pandas as pd

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


class RegClassifier1(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=8):
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
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=8): # 7아니면 8
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

class RNN_G(nn.Module):
    def __init__(self, input_size=42, hidden_size=16, num_layers=2, batch_size=20, num_classes=2):
        super(RNN_G, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.RNN = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        x = x.view(self.batch_size, -1, 400).transpose(2, 1)
        out, _ = self.RNN(x, self.h0)
        h_t = out[:, -1, :]
        out = self.fc1(h_t)
        return out    

# CnnLSTM

class CnnLSTM(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8, hidden_size=16, num_layers=2, batch_size=20, num_classes=8):
        super(CnnLSTM, self).__init__()
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

class CnnLSTM_MG(nn.Module):
    def __init__(self, num_channels=2, channel_1=16, channel_2=8, hidden_size=16, num_layers=2, batch_size=20, num_classes=2):
        super(CnnLSTM_MG, self).__init__()
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

        self.LSTM = nn.LSTM(64, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
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

class RegClassifierG(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=2):
        super(RegClassifierG, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(channel_1)
        self.bn2 = nn.BatchNorm2d(channel_2)
        
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
    
    
class RegClassifierAge2(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=8):
        super(RegClassifierAge2, self).__init__()
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

class CNN_G(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=2):
        super(CNN_G, self).__init__()
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

def check_accuracy(loader, model, dataset='valid', batch_size=128):
    dataset = dataset
    if dataset == 'train':
        print('Checking accuracy on train set')
    elif dataset == 'valid':
        print('Checking accuracy on validation set')
    elif dataset == 'test':
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    
    ac_gender = []
    gender_class_ac = []
    classes_gender = ['여성', '남성']
    correct_gender = list(0. for i in range(len(classes_gender)))
    total_gender = list(0. for i in range(len(classes_gender)))
    duration = []
    
    ac = []
    
    model.eval()
    with torch.no_grad():
        for dict in loader:
            x = dict['input']
            y = dict['label']
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            pred_start = time.perf_counter()
            scores = model(x)
            _, preds = scores.max(1)
            
            pred_end = time.perf_counter()
            duration.append(pred_end - pred_start)
                        
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            correct_num_gender = (preds == y).squeeze()
            
            for i in range(len(dict['label'])):
                label = y[i]
                correct_gender[label] += correct_num_gender[i].item()
                total_gender[label] += 1
            
        acc = float(num_correct) / num_samples
        
        ac.append(acc)
        
        print('Gender  : Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        for i in range(len(classes_gender)):
            gender_class_ac.append(100 * correct_gender[i] / total_gender[i])
            print("Accuracy of %4s : %2d %%" % (classes_gender[i], 100 * correct_gender[i] / total_gender[i]))
            
        print("\nDuration : %.6f초 (per input)\n" % (np.mean(duration) / batch_size))
        
        return ac, gender_class_ac


def tester(model, batch_size, model_type="CNN"):
    if model_type == "CNN":
        x = torch.zeros((batch_size, 3, 13, 400), dtype=dtype)  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    elif model_type == "RNN":
        x = torch.zeros((batch_size, 400, 39), dtype=dtype)  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    test_model = model
    scores = test_model(x)
    print(scores.size())



def train(model, loader_train, loader_valid, tb_path, optimizer, epochs=1, batch_size=512, print_every=500):
    model = model.to(device=device)
    writer = tb_path
    
    running_loss = 0.0
    
    acc = []
    acc_val = []
    loss_train = []
    
    time_log = []
    gender_class_train_ac = {'여성': [], '남성': []}
    gender_class_val_ac = {'여성': [], '남성': []}
    
    epoch_loss = 0.0
    
    gender_weight = torch.tensor([0.618, 1], dtype=torch.float32)
    gender_weight = gender_weight.to(device=device)

    for e in range(epochs):
        model.train()
        start = time.perf_counter()
        for t, dict in enumerate(loader_train):
            
            x = dict['input']
            y = dict['label']
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.int64)

            scores = model(x)
            crit = nn.CrossEntropyLoss(weight=gender_weight)
            loss = crit(scores, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            writer.add_scalar('training loss', running_loss / 10, epochs * len(loader_train) + t)
            running_loss = 0
            
            if t % print_every == 0:
                print('Iteration %d --- Train Loss = %.4f' % (t+1, loss.item()))
        end = time.perf_counter()
        print("Epoch %d finished --- Duration : %.4f초" % (e + 1, (end - start)))
        time_log.append(end - start)
        
        acc1, class_train = check_accuracy(loader=loader_train, model=model, dataset='train', batch_size=batch_size)
        acc_val1, class_val = check_accuracy(loader=loader_valid, model=model, dataset='valid', batch_size=batch_size)
        
        acc.append(acc1)
        acc_val.append(acc_val1)
        
        for i in range(len(list(gender_class_train_ac.keys()))):
            gender_class_train_ac[list(gender_class_train_ac.keys())[i]].append(class_train[i])
            gender_class_val_ac[list(gender_class_val_ac.keys())[i]].append(class_val[i])
        
        loss_train.append(loss.detach().cpu().item())
    
    df_gender_train = pd.DataFrame(gender_class_train_ac)
    df_gender_val = pd.DataFrame(gender_class_val_ac)
    
    import matplotlib.pyplot as plt
    plt.plot(acc,'-')
    plt.plot(acc_val,'-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Validation'])
    plt.title('Accuracy') 
    plt.show()
    
    plt.plot(loss_train)
    plt.ylabel('loss_train')
    plt.xlabel('epoch')
    plt.title('Model loss') 
    plt.show()
    
    df_gender_train.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.legend(['Female', 'Male'])
    plt.title("Gender Accuracy (Train)")
    plt.show()

    df_gender_val.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.legend(['Female', 'Male'])
    plt.title("Gender Accuracy (Validation)")
    plt.show()
    
    print("Mean Duration for Training per epoch = %.4f초\n" % (np.mean(time_log)))
   
    print('---- FINISHED!! ----')
