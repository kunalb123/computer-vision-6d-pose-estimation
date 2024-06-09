import torch.optim as optim
from model import DeepPose
from loss import CompositeLoss
import torch 
from dataloader import LineMODCocoDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
import argparse

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
if torch.backends.mps.is_available():
    device = torch.device('mps')
print('device being used:', device)


def save_checkpoint(state, filename="checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def load_model(checkpoint_path, model, optimizer=None):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Remove 'module.' prefix from state_dict keys if it exists
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    print("=> Loaded checkpoint")


def train_model(model, dataloader, loss_fn, optimizer, num_epochs=60, checkpoint_path='checkpoint.pth'):
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)
        print('Loaded checkpoint!')
    
    model.train()
    for epoch in tqdm(range(start_epoch, num_epochs)):
        running_loss = 0.0
        i = 0
        for images, target_dict in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            i += 1
            print(i)
            images = images.to(device)
            targets = target_dict['gt_maps'].to(device)

            gt_belief_maps = targets[:, :9, :, :]
            gt_vector_fields = targets[:, 9:, :, :]

            optimizer.zero_grad()

            pred_belief_maps, pred_vector_fields = model(images)

            loss = loss_fn(pred_belief_maps, gt_belief_maps, pred_vector_fields, gt_vector_fields)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('loss after batch:', loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': running_loss / len(dataloader)
        }
        save_checkpoint(checkpoint, filename=checkpoint_path)


class GaussianNoise(object):
    def __init__(self, mean=0.0, std=2.0):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        # Add Gaussian noise
        noise = torch.randn(tensor.size()) * self.std + self.mean
        tensor = tensor + noise
        return tensor


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training script parser")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--stages', type=int, default=5)
    parser.add_argument('--extra_conv', type=bool, default=False)
    args = parser.parse_args()

    print('TRAINING MODEL WITH PARAMS:')
    print('lr', args.lr)
    print('batch size', args.batch_size)
    print('stages', args.stages)
    print('extra conv', args.extra_conv)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print('device being used:', device)

    # Initialize model, loss, and optimizer
    model = DeepPose(extra_conv=args.extra_conv, num_final_stages=args.stages).to(device)
    composite_loss = CompositeLoss(stages=model.num_stages+1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # origial lr = 0.0001 (divide original learning rate by gpu and batchsize)
    # Train the model
    root = ''
    modelsPath = 'lm_models/models/models_info.json'

    OBJECT = 1
    
    annFile = f'train_annotations_obj{OBJECT}.json'
    checkpoint_path = f'obj{OBJECT}_checkpoint_epochs{args.epochs}_lr{args.lr}_batch_size{args.batch_size}_stages{args.stages}_extra_conv{args.extra_conv}.pth'
    print('will write to file:', checkpoint_path)

    dataset = LineMODCocoDataset(root, annFile, modelsPath)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)#, num_workers=4)
    train_model(model, dataloader, composite_loss, optimizer, checkpoint_path=checkpoint_path, num_epochs=args.epochs)