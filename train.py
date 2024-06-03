import torch.optim as optim
from model import DeepPose
from loss import CompositeLoss
import torch 
from dataloader import LineMODCocoDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
if torch.backends.mps.is_available():
    device = torch.device('mps')
print('device being used:', device)


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def train_model(model, dataloader, loss_fn, optimizer, num_epochs=60, checkpoint_path='checkpoint.pth.tar'):
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)
    
    model.train()
    for epoch in tqdm(range(start_epoch, num_epochs)):
        running_loss = 0.0
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            targets = targets.to(device)

            gt_belief_maps = targets[:, :9, :, :]
            gt_vector_fields = targets[:, 9:, :, :]

            optimizer.zero_grad()

            # Forward pass
            pred_belief_maps, pred_vector_fields = model(images)
            # print(type(pred_belief_maps))
            # print(type(gt_belief_maps))
            # print(type(pred_vector_fields))
            # print(type(gt_vector_fields))
            # Compute loss
            loss = loss_fn(pred_belief_maps, gt_belief_maps, pred_vector_fields, gt_vector_fields)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': running_loss / len(dataloader)
        }
        save_checkpoint(checkpoint, filename=checkpoint_path)


if __name__ == '__main__':

    # Initialize model, loss, and optimizer
    model = DeepPose().to(device)
    composite_loss = CompositeLoss(stages=6)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    root = 'test_data'
    modelsPath = 'lm_models/models/models_info.json'
    annFile = 'LOOKHEREannotations.json'
    dataset = LineMODCocoDataset(root, annFile, modelsPath)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
    train_model(model, dataloader, composite_loss, optimizer)
