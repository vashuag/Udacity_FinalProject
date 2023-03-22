#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import argparse
import json

# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = "Parser of training script")

parser.add_argument ('data_dir', help = 'Provide data directory. Mandatory argument', type = str)
parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', type = str)
parser.add_argument ('--arch', help = 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str)
parser.add_argument ('--lrn', help = 'Learning rate, default value 0.001', type = float)
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier. Default value is 2048', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int)
parser.add_argument ('--GPU', help = "Option to use GPU", type = str)

#setting values data loading
args = parser.parse_args ()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#defining device: either cuda or cpu
if args.GPU == 'GPU':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
else:
    device = 'cpu'

#data loading
if data_dir: 

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)



    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    #making sure we do have value for data_dir
    # Define your transforms for the training, validation, and testing sets
  
    #end of data loading block

#mapping from category label to category name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model (arch, hidden_units):

    num_classes = len(train_data.class_to_idx)

    if arch == 'vgg13': #setting model based on vgg13
        model = models.vgg13 (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: #if case hidden_units were given
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, num_classes)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: #if hidden_units not given
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (25088, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, num_classes)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    else: #setting model based on default Alexnet ModuleList
        arch = 'alexnet' #will be used for checkpoint saving, so should be explicitly defined
        model = models.alexnet (pretrained = True)

        for param in model.parameters():
            param.requires_grad = False
        
        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, num_classes)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: #if hidden_units not given
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (9216, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, num_classes)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    model.classifier = classifier #we can set classifier only once as cluasses self excluding (if/else)
    return model, arch



#loading model using above defined functiion
model, arch = load_model (args.arch, args.hidden_units)

#Actual training of the model
#initializing criterion and optimizer
criterion = nn.NLLLoss ()
if args.lrn: #if learning rate was provided
    optimizer = optim.Adam (model.classifier.parameters (), lr = args.lrn)
else:
    optimizer = optim.Adam (model.classifier.parameters (), lr = 0.001)


model.to (device) #device can be either cuda or cpu
#setting number of epochs to be run
if args.epochs:
    epochs = args.epochs
else:
    epochs = 8
steps = 0
running_loss = 0

print_every = 40

for epoch in range(epochs):
    
    for images,labels in trainloader:
        steps+=1
        
        images,labels = images.to(device) , labels.to(device)
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps,labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss +=loss.item()
        
        if steps % print_every ==0:
        
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                

                for images,labels in validloader:
                    images,labels = images.to(device) , labels.to(device)

                    log_ps = model(images)
                    loss = criterion(log_ps,labels)

                    valid_loss+=loss.item()

                    ps = torch.exp(log_ps)

                    top_ps,top_class = ps.topk(1,dim = 1)
                    equality = top_class ==labels.view(*top_class.shape)

                    accuracy +=torch.mean(equality.type(torch.FloatTensor)).item()
                
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy*100/len(validloader):.3f}")
            
            
            
            runnning_loss = 0
            
            model.train()

#saving trained Model
model.to ('cpu') #no need to use cuda for saving/loading model.
# Save the checkpoint



#creating dictionary for model saving
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'mapping':  train_data.class_to_idx,
              'optimizer_state_dict': optimizer.state_dict(),
              'num_epochs': epochs

             }
#saving trained model for future use
if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')
