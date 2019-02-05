import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader  # For custom data-sets
import numpy as np
import os
from PIL import Image
from numpy import array
import argparse

# PARAMETERS
num_epochs = 5
batch_size = 2
learning_rate = 0.001

# DATA PATHS
_INPUT_PATH ='./input/background'
_TRUTH_PATH ='./input/truth'
_PREDICT_PATH ='./input/to_predict'
_PREDICT_TRUTH_PATH ='./input/to_predict_truth'

# DEVICE CONFIGURATION
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.down11 = nn.Sequential(
            #nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #3= nb de channels (car image couleur), 32 = nombre de channels de sortie, kernel_size =3 taille du filtre de convolution (3x3), padding : 0 padding pour rester a "same" la meme taille d'image, padding 1 formule du padding voir sur internet mais pour simplifier c'est (kernel_size -1)/2
            nn.ReLU())
        self.down12 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.down21 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU())
        self.down22 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.down31 = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU())
        self.down32 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.down41 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU())
        self.down42 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU())
        self.down43 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU())
        self.down44 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU())
        self.down45 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU())
        self.down46 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU())
        self.down47 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU())
        self.down48 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.down51 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU())
        self.down52 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU())
        self.down53 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU())
        self.down54 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.up11 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1), #output_padding=(input_shape[0],input_shape[1]) ajouter ca si ca marche pas Tester si c'est bien la taille originale qu'il faut mettre | essayer avec output_padding = 1
            nn.ReLU())
        self.up12 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU())
        self.up21 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), #output_padding=(input_shape[0],input_shape[1]) ajouter ca si ca marche pas Tester si c'est bien la taille originale qu'il faut mettre | essayer avec output_padding = 1
            nn.ReLU())
        self.up22 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU())
        self.up31 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 96, 3, stride=2, padding=1, output_padding=1), #output_padding=(input_shape[0],input_shape[1]) ajouter ca si ca marche pas Tester si c'est bien la taille originale qu'il faut mettre | essayer avec output_padding = 1
            nn.ReLU())
        self.up32 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 3, stride=1, padding=1),
            nn.ReLU())
        self.up41 = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.ConvTranspose2d(96, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU())
        self.up42 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU())
        self.up43 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU())
        self.up44 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), #remettre 32 2 au lieu de 32 3
            nn.ReLU())
        self.up51 = nn.Sequential(
            nn.BatchNorm2d(6), #remettre batchnorm 2
            nn.ConvTranspose2d(6, 3, 3, stride=1, padding=1), #remettre 2 2 3 au lieu de 3 2 3
            nn.Sigmoid())

    def forward(self, input):
        #print('input : ', input.size())
        down11 = self.down11(input)
        down12 = self.down12(down11)
        down21 = self.down21(down12)
        down22 = self.down22(down21)
        down31 = self.down31(down22)
        down32 = self.down32(down31)
        down41 = self.down41(down32)
        down42 = self.down42(down41)
        down43 = self.down43(down42)
        down44 = self.down44(down43)
        down45 = self.down45(down44)
        down46 = self.down46(down45)
        down47 = self.down47(down46)
        down48 = self.down48(down47)
        down51 = self.down51(down48)
        down52 = self.down52(down51)
        down53 = self.down53(down52)
        down54 = self.down54(down53)
        up11 = self.up11(down54)
        up12 = self.up12(torch.cat([up11, down53],1))
        up21 = self.up21(up12)
        up22 = self.up22(torch.cat([up21, down47],1))
        up31 = self.up31(up22)
        up32 = self.up32(torch.cat([up31, down31],1))
        up41 = self.up41(up32)
        up42 = self.up42(up41)
        up43 = self.up43(up42)
        up44 = self.up44(up43)
        up51 = self.up51(torch.cat([up44,input],1))

        return up51

# CUSTOM DATASET CLASS
class CustomDataset(Dataset):

    def __init__(self,data_dir,mask_dir):

        self.data_dir=data_dir
        self.mask_dir=mask_dir

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self,idx):
        im=Image.open(os.path.join(self.data_dir,os.listdir(self.data_dir)[idx])).convert("RGB") #ex : im = l'image qui est dans data_dir/train_imageidx
        im_mask=Image.open(os.path.join(self.mask_dir,os.listdir(self.data_dir)[idx].split(".")[0] #ex : mask_dir/train_image[premier caractere apres le point]_mask.gif
                                     + '_mask.png')).convert("RGB") #L for grayscale

        if(str(device) == 'cpu'): #if CPU is used
            sample={'image':self.to_tensor(array(im)),'mask':self.to_tensor(array(im_mask))}
        else: #if GPU is used
            sample={'image':self.to_tensor(array(im)).cuda(),'mask':self.to_tensor(array(im_mask)).cuda()}

        return sample

# TRAINING FUNCTION
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0
    for i,data in enumerate(train_loader):
        image = data['image'] # converting NHWC to NCHW (number of samples, number of channels, height, width)
        mask = data['mask']

        optimizer.zero_grad()

        output = model(image)

        loss = criterion(output, mask)
        loss.backward()

        optimizer.step()

        #Statistics
        print ('Step [{}/{}] | Loss = {}'.format(i+1,len(train_loader),loss), end="\r")
        running_loss+=loss
        if (i == len(train_loader)-1): #last step of the epoch
            print('Total loss = {}\nMean loss = {}\n'.format(running_loss, running_loss/len(train_loader)))

# TESTING THE MODEL
def test(model, test_loader, criterion):
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for i,data in enumerate(test_loader):
            image = data['image']
            mask = data['mask']

            output = model(image)

            loss = criterion(output, mask)

            #Statistics
            print ('Image [{}/{}] | Loss = {}'.format(i+1,len(test_loader),loss), end="\r")
            test_loss+=loss
        print('Total loss = {}\nMean loss = {}\n'.format(test_loss.item(), test_loss.item()/len(test_loader)))

def main():

    # PARSING COMMAND
    train_mode = False #default mode

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t","--train", help="train the network", action="store_true")
    group.add_argument("-p","--predict", help="predict some masks", action="store_true")
    args = parser.parse_args()
    if args.train:
        train_mode = True
    if args.predict:
        train_mode = False

    # CREATING CUSTOM DATASET AND LOADING DATAS
    train_set=CustomDataset(_INPUT_PATH,_TRUTH_PATH)
    train_loader=DataLoader(train_set,batch_size=batch_size//2,shuffle=True) #//2, otherwise train_loader will divide its number of element by 2

    test_set=CustomDataset(_PREDICT_PATH,_PREDICT_TRUTH_PATH)
    test_loader=DataLoader(test_set,batch_size=batch_size//2) #test_set is an unordered set so it is not possible to pick to_predict images sequentially

    # CREATING THE MODEL
    model = ConvNet().to(device)

    print('\n-------DEEPSEG-------\nUsing device ', device)
    print('Running in ', 'training ' if train_mode else 'predict ','mode\n---------------------\n')

    # TRAIN MODE
    if train_mode:

        # DEFINING LOSS AND OPTIMIZER
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # TRAINING THE MODEL
        for epoch in range(0, num_epochs):
            print('Epoch ', epoch+1, '/', num_epochs, '\n--------------')
            train(model, train_loader, criterion, optimizer, epoch)

        # SAVING THE MODEL CHECKPOINT
        torch.save(model.state_dict(), 'cnn1.pth')

    # PREDICT / EVALUATION MODE
    else:
        criterion = nn.MSELoss()
        model.load_state_dict(torch.load('cnn1.pth')) #Loading checkpoint file
        test(model, test_loader, criterion)

if __name__ == "__main__":
    main()
