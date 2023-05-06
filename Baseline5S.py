import ModelFiles
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import pathlib
import glob

if __name__ == '__main__':
    #setup model
    model = ModelFiles.ConvNet(5)
    model = model.to('cpu')
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    #setup transformer
    transformer = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    #load data

    train_path = "/Users/jbalk/Documents/ML_Data/FinalProject/5_Shot/LSTM_Training/T0/D_Train"
    test_path = "/Users/jbalk/Documents/ML_Data/FinalProject/5_Shot/LSTM_Training/T0/D_Test"

    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transformer),
        batch_size=1, shuffle=False, num_workers=8
    )

    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transformer),
        batch_size=1, shuffle=False, num_workers=8
    )

    #data analytics

    root = pathlib.Path(train_path)
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
    n = len(classes)

    print(classes)
    print(n)

    train_count = len(glob.glob(train_path+'/**/*.jpg'))
    test_count = len(glob.glob(test_path+'/**/*.jpg'))

    print(train_count)
    print(test_count)

    #assess model

    best_accuracy = 0.0
    num_epoch = 50

    for epoch in range(num_epoch):

        model.train()
        train_accuracy = 0.0
        train_loss = 0.0

        for i, (images,labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data*images.size(0)
            _,prediction = torch.max(outputs.data,1)
            train_accuracy += int(torch.sum(prediction==labels.data))
        
        train_accuracy = train_accuracy/train_count
        train_loss = train_loss/train_count

        model.eval()
        test_accuracy = 0.0
        for i, (images,labels) in enumerate(test_loader):
            outsputs = model(images)
            _,prediction = torch.max(outputs.data, 1)
            test_accuracy += int(torch.sum(prediction==labels.data))
        
        test_accuracy = test_accuracy/test_count

        print("Epoch %d Complete\n" % epoch)
        print("Training Accuracy: %0.4f" % train_accuracy)
        print("Training Loss: %0.4f" % train_loss)
        print("Testing Accuracy: %0.4f" % test_accuracy)
        print("---------------\n")

        if(test_accuracy > best_accuracy):
            best_accuracy = train_accuracy
            torch.save(model.state_dict(), 'best.model')

    #print model outputs
    print("Best accuracy attained:")
    print(best_accuracy)
