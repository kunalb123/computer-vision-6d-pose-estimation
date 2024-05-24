import torch.optim as optim

# Initialize model, loss, and optimizer
model = DeepPose().to(device)
composite_loss = CompositeLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_model(model, dataloader, loss_fn, optimizer, num_epochs=60):
    model.train()
    for epoch in range(num_epochs):
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

# Train the model
train_model(model, dataloader, composite_loss, optimizer)
