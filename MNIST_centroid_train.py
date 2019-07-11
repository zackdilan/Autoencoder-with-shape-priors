import os
import random
import time
import cv2
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torchvision.datasets as dataset
from torch.utils.data import  DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image,ImageStat
from visdom import Visdom

# load MNIST dataset
root = "/home/robin/Thesis/Autoencoder/MNIST1_data"
if not(os.path.exists(root)):
    os.mkdir(root)

class MNIST1(dataset.MNIST):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target,mean_pixel) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        ##afine transform acording a randome offset
        def affine_trans(img):
            image = torch.squeeze(img,0)
            rnumber = [random.uniform(0,3) for x in range(2)]
            #returns a torch tensor of rnumber..
            rnumber = torch.FloatTensor(rnumber)
            x_offset,y_offset = rnumber
            M = np.float32([[1,0,x_offset],[0,1,y_offset]])
            rows,cols = rows,cols = image.shape[0:]
            result = cv2.warpAffine(image.numpy(),M,(rows,cols))
            result = torch.from_numpy(result).float()
            return rnumber,result


        if self.transform is not None:
            img = self.transform(img)
            new_img = torch.squeeze(img,0)
            rnumber,result = affine_trans(img)
            centroid = list(ndimage.measurements.center_of_mass(new_img.numpy()))
            centroid = torch.FloatTensor(centroid)

        if self.target_transform is not None:
            target = self.target_transform(target)

        values = {"image":img,"target":target,
                 "centroid":centroid,"randnum":rnumber,"warp_img":result}
        #return img,target,centroid_x,centroid_y,rnumber,result
        return values

trans = transforms.Compose([transforms.ToTensor()])

# if not exist, download mnist dataset

train_set = MNIST1(root=root, train=True, transform=trans, download=True)
test_set = MNIST1(root=root, train=False, transform=trans, download=True)

batch_size = 256
train_loader = DataLoader(dataset=train_set,
                 batch_size=batch_size,
                shuffle=True,pin_memory=False)
test_loader = DataLoader(dataset=test_set,
                 batch_size=batch_size,
                shuffle=True,pin_memory=False)
#create a dataloader dictionary
dataloaders = {"train":train_loader,"val":test_loader}

print(len(dataloaders["train"]))
print ('==>>> total training batch number: {}'.format(len(train_loader)))
print ('==>>> total testing batch number: {}'.format(len(test_loader)))
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder =  nn.Sequential(
                        nn.Linear(28*28,256),
                        nn.ReLU(True),
                        nn.Linear(256,128),
                        nn.ReLU(True),
                        nn.Linear(128,64),nn.ReLU(True),
                        nn.Linear(64,20)
                        )
        self.decoder =  nn.Sequential(
                        nn.Linear(20,64),nn.ReLU(True),
                        nn.Linear(64,128),nn.ReLU(True),
                        nn.Linear(128,256),nn.ReLU(True),
                        nn.Linear(256,28*28),nn.Tanh()
                        )
    def forward(self,x,offset):
        encoded = self.encoder(x)
        decoded = torch.add(offset,encoded)
        decoded = self.decoder(decoded)
        return encoded,decoded



class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

#training loop
def train_val(model,criterion,optimizer,num_epochs):
        best_loss = 1000.0

        for epoch in range(num_epochs):
            print("Epoch {}/{}". format(epoch,num_epochs-1))
            print("-"*30)
            since = time.time()
            running_loss = 0.0
            for phase in ["train","val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                loss = 0.0
                reconstruction_loss = 0.0
                centroid_loss = 0.0

                for index,sampled_batch in enumerate (dataloaders[phase]):
                    image,warp_image =sampled_batch["image"],sampled_batch["warp_img"]
                    inputs = image.view(image.size(0),-1).to(device)     #shape [batch_size*784]
                    shifted_inputs = warp_image.view(warp_image.size(0),-1).to(device)   #shape [batch_size*784]
                    offset = sampled_batch["randnum"]   #shape [batch_size*2]
                    #converted to the dimension of encoded o/p from model to get added.
                    offset = F.pad(input=offset, pad=(18, 0, 0, 0), mode='constant', value=0)
                    centroid = sampled_batch["centroid"]   #shape [batch_size*2]
                    ##since centre of mass (centroid return the y centroid first, we swap the
                    #columns of centroid tensor..)
                    centroid = torch.index_select(centroid, 1, torch.LongTensor([1,0]))

                    #Forward

                    with torch.set_grad_enabled(phase == 'train'):
                        encode_out,output = model(inputs,offset.to(device))
                        #make encode out to compute the loss between the centroid #shape [batch_size*2]
                        centroid_out = encode_out[:,18:20]
                        centroid_loss = criterion(centroid_out,centroid.to(device))
                        #including two stage training strategy 1. Train the base autoencoder for few
                        #epochs to learn the basic reconstruction task.2.Then train for the centroid shift..
                        if epoch <2:
                            print("first stage")
                            reconstruction_loss =criterion(output,inputs)
                            loss = reconstruction_loss
                        else:
                            print("second stage")
                           #reconstruction loss between outputs and warped images
                            reconstruction_loss = criterion(output,shifted_inputs.to(device))
                             #Total loss ..
                            loss = centroid_loss+reconstruction_loss 
                        
                        ###backward
                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    #loss metrics
                    running_loss+=loss.item()
                    epoch_loss = loss.item()
                #epoch_loss = running_loss

                if phase == 'train':
                    plotter.plot('Total loss', 'train', ' Loss', epoch, epoch_loss)
                    plotter.plot("reconstruction_loss", "train","recon_loss",epoch,reconstruction_loss.item())
                    plotter.plot("centroid_loss", "train","centroid_loss",epoch,centroid_loss.item())
                else:
                    plotter.plot('Total loss', 'validation', ' Loss', epoch, epoch_loss)
                    plotter.plot("reconstruction_loss", "validation","recon_loss",epoch,reconstruction_loss.item())
                    plotter.plot("centroid_loss", "validation","centroid_loss",epoch,centroid_loss.item())

                print("{} --- Epoch {}, Epoch  loss : {:.4f} ,".format(phase ,epoch,epoch_loss))
                # saving the model
                #if epoch %10 ==0:
                if phase == "val" and epoch_loss<best_loss:
                        best_loss = epoch_loss
                        torch.save({ "epoch": epoch,
                            "model_state_dict":model.state_dict(),
                            "optimizer_state_dict":optimizer.state_dict(),
                            "loss":epoch_loss,
                            },'/home/robin/Thesis/Autoencoder/logs/centroid_auto2/train_centroid2_epoch{}.pth'.format(epoch))
                else:
                        pass

                time_elapsed = time.time()-since
                print("Epoch complete in {:.0f}min - {:.0f}secs".format(time_elapsed/60,time_elapsed%60))
if __name__ == "__main__":
    num_epochs = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    criterion= nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    #optimizer = torch.optim.Adadelta(model.parameters(),lr = 1.0,weight_decay=0.0)
    plotter = VisdomLinePlotter(env_name='Centroid Train')
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size = 10,gamma = 0.1)
    train_val(model,criterion,optimizer,num_epochs)
