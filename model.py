# idk if this actually works or not lol run at your own peril

import torch as torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import math
import os

'''
============================================================
                    UTILITY FUNCTIONS
============================================================
'''

# create mask for bounding boxes
# not really used but im keeping it if we need to do transforms in the future
def create_mask(bb, path):
    in_img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    rows,cols,*_ = in_img.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def create_bb_array(row):
    return np.array([row.iloc[4], row.iloc[6], row.iloc[5], row.iloc[7]])

# functions to show image + bounding box
def create_bound_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[0], bb[2]), bb[1]-bb[0], bb[3]-bb[2], color=color, fill=False, lw=3)

def show_bound_box(im, bb_pred, bb_actual, ax=None):
    ax=ax
    ax.imshow(im)
    ax.add_patch(create_bound_rect(bb_actual))
    ax.add_patch(create_bound_rect(bb_pred, color='blue'))
    
'''
============================================================
                        DATA LOADING
============================================================
'''
# dataset class
class image_data(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        print("create data class")
        annotations = pd.read_csv(self.root_dir+'_annotations_merged.csv')
        class_dict = {'well done': 0, 'medium well done': 1, 'medium': 2, 'medium rare': 3, 'rare': 4, 'blue rare': 5}
        annotations['class'] = annotations['class'].apply(lambda x:  class_dict[x])
        bound_boxes = []
        for idx, row in annotations.iterrows():
            bb = create_bb_array(row)
            bound_boxes.append(bb)
        annotations['bboxes'] = bound_boxes
        annotations.reset_index()

        # other stuff?

        # define transforms
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5557, 0.4393, 0.3785],
                    std=[0.2852, 0.2857, 0.2876]
                ),
        ])

        self.filenames = annotations['filename'].values
        self.bboxes = annotations['bboxes'].values
        self.Y = annotations['class'].values

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        path = self.filenames[index]
        x = cv2.imread(self.root_dir+path).astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
        # x = self.transform(x)
        x = np.rollaxis(x, 2)
        bb = self.bboxes[index]
        y = self.Y[index]
        return x, bb, y

# define Residual block for resnet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(25088, num_classes)
        self.fc_bb = nn.Linear(25088, 4)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return self.fc(x), self.fc_bb(x)

'''
============================================================
                    RUNNING THE MODEL
============================================================
'''

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.multiprocessing.freeze_support()
    num_classes = 6
    num_epochs = 50
    batch_size = 16
    max_learning_rate = 0.01
    learning_rate = 0.0001 # initial
    split = 0.8
    dataset = image_data('./detect_steak.v2i.tensorflow/images/')
    # print(len(dataset))
    train_set, test_set = random_split(dataset, [math.floor(split*len(dataset)), math.ceil((1-split)*len(dataset))])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # load model
    # path to model (if it exists)
    PATH = './steak_net.pth'
    model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes).to(device)

    # TODO: IMPORTANT!!!
    # change to true if you want to regenerate the thing
    rerun = False
    if rerun or not os.path.exists(PATH):
        print("begin rerun")

        # loss func and optimizer
        criterion_class = nn.CrossEntropyLoss()
        criterion_bbox = nn.L1Loss(reduction='none')
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.002, momentum=0.9)

        step_size=(len(train_loader)/batch_size)*5
        print(step_size)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=max_learning_rate, step_size_up=step_size, mode='triangular2')
        train_loss = []

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            print("epoch: %d of %d" % (epoch+1, num_epochs))
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, bboxes, labels = data
                inputs = inputs.to(device)
                bboxes = bboxes.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output_class, output_bbox = model(inputs)
                loss_class = criterion_class(output_class, labels)
                loss_bbox = criterion_bbox(output_bbox, bboxes).sum(1)
                loss_bbox = loss_bbox.sum()
                # might be scuffed
                loss = loss_class + loss_bbox/1000.0
                loss.backward()
                optimizer.step()
                scheduler.step()

                # print statistics
                running_loss += loss.item()
                train_loss.append(loss.item())
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

            # TODO: uncomment if we bother to do a validation set
            # with torch.no_grad():
            #     correct = 0
            #     total = 0
            #     for images, bboxes, labels in valid_loader:
            #         images = images.to(device)
            #         bboxes = bboxes.to(device)
            #         labels = labels.to(device)
            #         outputs, output_bbox = model(images)
            #         _, predicted = torch.max(outputs.data, 1)
            #         total += labels.size(0)
            #         correct += (predicted == labels).sum().item()
            #         del images, labels, outputs
            #     print('Accuracy of the network on the {} validation images: {} %'.format(len(test_loader), 100 * correct / total))
        
        # TODO: figure out super convergence
        # run one more time with massively decreased learning rate
        # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate/100, max_lr=max_learning_rate/100, step_size_up=step_size, mode='triangular2')
        # for i, data in enumerate(train_loader, 0):
        #     # get the inputs
        #     inputs, bboxes, labels = data
        #     inputs = inputs.to(device)
        #     bboxes = bboxes.to(device)
        #     labels = labels.to(device)

        #     # zero the parameter gradients
        #     optimizer.zero_grad()

        #     # forward + backward + optimize
        #     output_class, output_bbox = model(inputs)
        #     loss_class = criterion_class(output_class, labels)
        #     loss_bbox = criterion_bbox(output_bbox, bboxes).sum(1)
        #     loss_bbox = loss_bbox.sum()
        #     # might be scuffed
        #     loss = loss_class + loss_bbox/1000.0
        #     loss.backward()
        #     optimizer.step()
        #     scheduler.step()

            # print statistics
            running_loss += loss.item()
            train_loss.append(loss.item())
            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        print('Finished Training')
        train_legend = mlines.Line2D([], [], color='blue', label='Train Loss')
        plt.plot(train_loss, label="Train Loss")
        plt.legend(handles=[train_legend])
        plt.savefig('loss.png')
        # plt.show()

        torch.save(model.state_dict(), PATH)
    
    else:
        print("loading from file")
        model.load_state_dict(torch.load(PATH))
        model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, bboxes, labels in test_loader:
            images = images.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)
            outputs, output_bbox = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print image bounding boxes by batch
            # fig, axs = plt.subplots(ncols=4, nrows=4)
            # for i in range(len(labels)):
            #     img = images[i].detach().cpu().numpy()
            #     img = np.rollaxis(img, 0, 3)
            #     bbox_orig = bboxes[i].detach().cpu().numpy()
            #     bbox_out = output_bbox[i].detach().cpu().numpy()
            #     show_bound_box(img, bbox_out, bbox_orig, axs[int(i/4)][i%4])
            # plt.show()

            del images, labels, outputs
        print('Accuracy of the network on the {} validation images: {} %'.format(len(test_loader)*batch_size, 100 * correct / total))


# main function so it doesn't throw a multithread error
if __name__ == '__main__':
    run()
    