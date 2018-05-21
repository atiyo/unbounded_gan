import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
import numpy as np

num_epochs = 100
batch_size = 50
#input size of noise to the generator
noise_in_len = 100
#use cuda, or not?
use_cuda = True
#number of training steps of the discriminator before the generator is trained
k = 1
#image_save_freq is the number of batches between image saves. making this
#large (60k or above) means that only one image per epoch will be saved.
image_save_freq = 60000
print_step = 50

#helper function to check if fashion mnist exists and to download it if it doesn't.
def get_fashion_mnist():
    if (not os.path.exists('./fashion_mnist/processed/test.pt')) or (not os.path.exists('./fashion_mnist/processed/training.pt')):
        if os.path.isdir('./fashion_mnist'):
            os.system('rm -rf ./fashion_mnist')
            os.system('mkdir ./fashion_mnist')
        else:
            os.system('mkdir ./fashion_mnist')
        torchvision.datasets.FashionMNIST('./fashion_mnist', download=True)

#grab fashion_mnist if it's not already downloaded.
get_fashion_mnist()
#data set for fashion_mnist which also preprocesses images to the range [-1,1]
class image_set(Dataset):
    def __init__(self, data=torchvision.datasets.FashionMNIST('./fashion_mnist')):
        self.data = data
        self.process_pil = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        image = self.process_pil(image)
        image = image - 0.5
        image = 2 * image
        return image

#generator definition
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc_1 = nn.Linear(100, 1024)
        self.fc_2 = nn.Linear(1024, 128 * 7 * 7)
        self.conv_3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.bn_3 = nn.BatchNorm2d(64)
        self.conv_4 = nn.Conv2d(16, 1, 5, stride=1, padding=2)

    def forward(self, noise):
        out = self.fc_1(noise)
        out = F.leaky_relu(out, 0.2)
        out = self.fc_2(out)
        out = F.leaky_relu(out, 0.2)
        out = out.view([-1, 128, 7, 7])
        out = F.pixel_shuffle(out, 2)
        out = self.conv_3(out)
        out = self.bn_3(out)
        out = F.leaky_relu(out, 0.2)
        out = F.pixel_shuffle(out, 2)
        out = self.conv_4(out)
        out = F.tanh(out)
        return out

#discriminator definition
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(1, 64, 5, stride=1, padding=2)
        self.conv_2 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.fc_3 = nn.Linear(128*14*14, 1)

    def forward(self, img):
        out = self.conv_1(img)
        out = F.leaky_relu(out, 0.2)
        out = F.max_pool2d(out, 2)
        out = self.conv_2(out)
        out = F.leaky_relu(out, 0.2)
        out = out.view([-1,128*14*14])
        out = self.fc_3(out)
        out = F.sigmoid(out)
        return out

#transform torch tensors from [0,1] back to a PIL image.
deprocess_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])

#function which grabs a batch of tensors and saves them
def save_output_plot(pytorch_variable, index, path='./l1_output/'):
    num_images = pytorch_variable.shape[0]
    data = pytorch_variable.data.view([-1,1,28,28])
    data = data / 2.
    data = data + 0.5
    output = torchvision.utils.make_grid(data, nrow=25)
    output = deprocess_pil(output)
    file_name = path+'output_{}.jpg'.format(index)
    output.save(file_name)

if __name__=='__main__':

    #create an output directory if one doesn't exist
    if not os.path.isdir('./l1_output'):
        os.system('mkdir ./l1_output')

    #set up a constant batch of noise to track training progress of the generator
    if use_cuda:
        plot_noise = Variable(torch.randn((200, noise_in_len)).cuda())
    else:
        plot_noise = Variable(torch.randn((200, noise_in_len)))

    #load the data set and create an iterator
    data = image_set()
    image_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)

    #set up the networks and bind them to the GPU if applicable
    G = Generator()
    D = Discriminator()

    if use_cuda:
        G.cuda()
        D.cuda()

    #optimizers for both networks
    g_optimizer = torch.optim.RMSprop(G.parameters(), lr=5e-5)
    d_optimizer = torch.optim.RMSprop(D.parameters(), lr=5e-5)

    #plot_index is used to construct sequential output names for generated images.
    plot_index = 0
    for epoch in range(num_epochs):
        for step, data in enumerate(image_loader):
            #assign noise and images to train the discriminator
            image = data
            if use_cuda:
                image = Variable(image.cuda())
                noise = Variable(torch.randn((batch_size, noise_in_len)).cuda())
            else:
                image = Variable(image)
                noise = Variable(torch.randn((batch_size, noise_in_len)))
            
            #train on real data
            d_optimizer.zero_grad()
            d_loss = torch.mean(torch.abs((D(image) - 1)))
            d_loss.backward()
            d_optimizer.step()

            #train on generated data
            d_optimizer.zero_grad()
            d_loss = torch.mean(torch.abs(D(G(noise))))
            d_loss.backward()
            d_optimizer.step()

            # #clamp the weights in the discriminator to try and force it to be Lipschitz.
            # #there are likely better ways to force D to be Lipschitz, but this
            # #works in the current scenario.
            # for p in D.parameters():
                # p.data.clamp_(-0.01, 0.01)

            #give the discriminator a 100 batch training headstart, then train
            #the generator every k batches.
            if step % k == 0:
                #noise to train the generator
                if use_cuda:
                    noise = Variable(torch.randn((batch_size, noise_in_len)).cuda())
                else:
                    noise = Variable(torch.randn((batch_size, noise_in_len)))

                #train the generator
                g_optimizer.zero_grad()
                g_loss = torch.mean(torch.abs(D(G(noise))-1))
                g_loss.backward()
                g_optimizer.step()

            #save the output every image_save_freq batches, or every epoch
            #(whichever is more frequent)
            if step % print_step == 0:
                print('Epoch: {}. Step: {}. Generator Loss: {}. Discriminator Loss: {}'.format(epoch, step, g_loss.data.cpu().numpy(), d_loss.data.cpu().numpy()))

            if step % image_save_freq == 0:
                preds = G(plot_noise)
                preds = preds.cpu()
                save_output_plot(preds,plot_index)
                plot_index += 1

    #clean up any mess we're leaving on the gpu
    if use_cuda:
        torch.cuda.empty_cache()


