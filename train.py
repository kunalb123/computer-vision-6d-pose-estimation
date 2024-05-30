import torch.optim as optim
from model import DeepPose
from loss import CompositeLoss
import torch 
from dataloader import LineMODCocoDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
if torch.backends.mps.is_available():
    device = torch.device('mps')
print('device being used:', device)

# Initialize model, loss, and optimizer
model = DeepPose().to(device)
composite_loss = CompositeLoss(stages=6)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_model(model, dataloader, loss_fn, optimizer, num_epochs=60):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            gt_belief_maps = targets[:, :9, :, :]
            gt_vector_fields = targets[:, 9:, :, :]

            optimizer.zero_grad()

            # Forward pass
            pred_belief_maps, pred_vector_fields = model(images)

            # Compute loss
            loss = loss_fn(pred_belief_maps, gt_belief_maps, pred_vector_fields, gt_vector_fields)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

if __name__ == '__main__':

    # Train the model
    root = 'test_data'
    modelsPath = 'lm_models/models/models_info.json'
    annFile = 'LOOKHEREannotations.json'
    dataset = LineMODCocoDataset(root, annFile, modelsPath)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    train_model(model, dataloader, composite_loss, optimizer)
