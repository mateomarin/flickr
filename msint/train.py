import torch
import torch.nn as nn
import torch.optim as optim
from mnist_loader import four_digit_train, four_digit_test
from transformer import VisionTransformer
from torch.utils.data import DataLoader

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    # Move model to device
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct_digits = 0
        total_digits = 0
        
        for images, labels in train_loader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)  # Shape: (batch_size, 4)
            
            # Forward pass
            digit_predictions = model(images)  # List of 4 tensors, each (batch_size, 10)
            
            # Calculate loss for each digit
            loss = 0
            for i in range(4):
                loss += criterion(digit_predictions[i], labels[:, i])
            loss = loss / 4  # Average loss across digits
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            
            # Calculate accuracy
            for i in range(4):
                pred = digit_predictions[i].argmax(dim=1)
                correct_digits += (pred == labels[:, i]).sum().item()
                total_digits += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_digits / total_digits
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct_digits = 0
        total_digits = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                digit_predictions = model(images)
                
                # Calculate validation loss
                loss = 0
                for i in range(4):
                    loss += criterion(digit_predictions[i], labels[:, i])
                val_loss += loss.item() / 4
                
                # Calculate accuracy
                for i in range(4):
                    pred = digit_predictions[i].argmax(dim=1)
                    correct_digits += (pred == labels[:, i]).sum().item()
                    total_digits += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_digits / total_digits
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n')
        
        # Add scheduler step at the end of each epoch
        scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader = DataLoader(four_digit_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(four_digit_test, batch_size=32)
    
    # Initialize model
    model = VisionTransformer()
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=10, device=device) 