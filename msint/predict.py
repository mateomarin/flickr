import torch
from mnist_loader import four_digit_test
from transformer import VisionTransformer

def predict_and_show(model, image, true_labels=None):
    # Prepare model for evaluation
    model.eval()
    
    # Add batch dimension and move to model's device
    image = image.unsqueeze(0).to(next(model.parameters()).device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(image)
        # Get digit predictions
        digit_predictions = [pred.argmax(dim=1)[0].item() for pred in predictions]
    
    print(f"Predicted digits: {digit_predictions}")
    if true_labels is not None:
        print(f"True digits:     {true_labels.tolist()}")
    print("-" * 40)

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    model = VisionTransformer()
    model.load_state_dict(torch.load('trained_model.pth'))
    model = model.to(device)
    
    # Get test samples
    test_loader = torch.utils.data.DataLoader(four_digit_test, batch_size=1, shuffle=True)
    
    print("\nPredictions:")
    print("=" * 40)
    # Show predictions for 10 random samples
    for i, (image, labels) in enumerate(test_loader):
        if i >= 10:  # Show 10 samples
            break
        print(f"Sample {i+1}:")
        predict_and_show(model, image[0], labels[0]) 