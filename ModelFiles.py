import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import math
import glob

#Convolutional Neural Network used for classification
class ConvNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNet, self).__init__()

        bnf = 32
        pks = 2
        ks = 3
        inc = 3

        self.imgsz = 256
        self.fsz = math.floor(self.imgsz/16)
        self.flat_count = (self.fsz**2)*32

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=32, kernel_size=ks)
        self.bn1 = nn.BatchNorm2d(num_features=bnf)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=pks)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=ks)
        self.bn2 = nn.BatchNorm2d(num_features=bnf)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=pks)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=ks)
        self.bn3 = nn.BatchNorm2d(num_features=bnf)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=pks)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=ks)
        self.bn4 = nn.BatchNorm2d(num_features=bnf)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=pks)

        self.fc = nn.Linear(in_features=self.flat_count, out_features=num_classes)
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        #fully connected layer
        x = torch.reshape(x, (-1,))
        x = self.fc(x)
        x = self.sm(x)

        return x

#function to set up and return CNN model
def getModel(n_class = 5, device = 'cpu'): #default is 5 class classification and 
    model = ConvNet(num_classes=n_class)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    return model, loss_function, optimizer #return model, loss function, and ADAM optimizer

#Meta-learner class
class MetaLearn(nn.Module):
    def __init__(self):
        super(MetaLearn, self).__init__()
        
        #define number of datasets used for each stage
        self.train_ct = 80
        self.test_ct = 20
        self.val_ct = 25

        #define my own gradient function
        self.grad = optim.Adam(self.parameters())

        #define other parameters
        self.batch_sz = 1
        self.rtpth = "/Users/jbalk/Documents/ML_Data/FinalProject/1_Shot/"
        self.pthex = "LSTM_Train/" #in training mode by default

        self.ind = 0

        #set up transformer
        self.transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def train_mode(self): #set model to evaluation mode
        self.pthex = "LSTM_Train/"
        self.ind = 0 #reset enumeration index
    
    def eval_mode(self): #set model to evaluation mode
        self.pthex = "LSTM_Test/"
        self.ind = 0 #reset enumeration index
    
    def validate_mode(self): #set model to evaluation mode
        self.pthex = "LSTM_Validation/"
        self.ind = 0 #reset enumeration index

    def forward(self):
        cnn_model, cnn_loss, cnn_opt = getModel(n_class=5)
        cnn_model.parameters = self.parameters()

        train_path = self.rtpth+self.pthex+"T"+str(self.ind)+"/D_train/"
        test_path = self.rtpth+self.pthex+"T"+str(self.ind)+"/D_test/"

        #run sub learner

        train_loader = DataLoader(
            torchvision.datasets.ImageFolder(train_path, transform=self.transformer),
            batch_size=1, shuffle=False, num_workers=8
        )

        test_loader = DataLoader(
            torchvision.datasets.ImageFolder(test_path, transform=self.transformer),
            batch_size=1, shuffle=False, num_workers=8
        )

        train_count = len(glob.glob(train_path+'/**/*.jpg'))
        test_count = len(glob.glob(test_path+'/**/*.jpg'))

        cnn_model.train()
        train_acc = 0.0
        train_loss = 0.0

        for i, (images,labels) in enumerate(train_loader):
            cnn_opt.zero_grad()
            outputs = cnn_model(images)
            loss = cnn_loss(outputs, labels)
            loss.backward()
            cnn_opt.step()

            train_loss += loss.cpu().data*images.size(0)
            _,prediction = torch.max(outputs.data,1)
            train_acc += int(torch.sum(prediction==labels.data))
        
        train_acc = train_acc/train_count
        train_loss = train_loss/train_count

        cnn_model.eval()
        test_acc = 0.0
        for i, (images,labels) in enumerate(test_loader):
            outputs = cnn_model(images)
            loss = cnn_loss(outputs, labels)
            _,prediction = torch.max(outputs.data, 1)
            test_acc += int(torch.sum(prediction==labels.data))

        test_acc = test_acc/test_count

        #index data index
        self.ind = self.ind + 1

        #update my parameters
        self.grad.step()
        self.parameters = loss*self.grad
        return loss #return the loss to be used in testing and validation modes

