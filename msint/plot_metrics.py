import matplotlib.pyplot as plt
import json

def plot_training_curves():
    # Load metrics from file
    with open('training_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot losses
    ax1.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, metrics['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, metrics['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    plot_training_curves() 