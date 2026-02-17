import time
import json
from collections import defaultdict

class TrainingTracker:
    def __init__(self, save_dir="training_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.epoch_times = []
        self.start_time = None
        
    def start_training(self):
        self.start_time = time.time()
        
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Log metrics for one epoch"""
        epoch_time = time.time() - self.start_time
        
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['learning_rate'].append(lr)
        self.metrics['time'].append(epoch_time)
        
        # Print to console
        print(f"Epoch {epoch}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} | "
              f"LR={lr:.6f} | Time={epoch_time:.2f}s")
        
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = os.path.join(self.save_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
        print(f"Metrics saved to {metrics_file}")
        
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
        
        epochs = self.metrics['epoch']
        
        # Loss curves
        axes[0, 0].plot(epochs, self.metrics['train_loss'], 
                       label='Train Loss', color='#1E88E5', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics['val_loss'], 
                       label='Val Loss', color='#E53935', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontweight='bold')
        axes[0, 0].set_ylabel('Loss', fontweight='bold')
        axes[0, 0].set_title('Training & Validation Loss', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.metrics['train_acc'], 
                       label='Train Acc', color='#1E88E5', linewidth=2)
        axes[0, 1].plot(epochs, self.metrics['val_acc'], 
                       label='Val Acc', color='#E53935', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
        axes[0, 1].set_title('Training & Validation Accuracy', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.metrics['learning_rate'], 
                       color='#43A047', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontweight='bold')
        axes[1, 0].set_ylabel('Learning Rate', fontweight='bold')
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training time
        axes[1, 1].plot(epochs, self.metrics['time'], 
                       color='#FB8C00', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontweight='bold')
        axes[1, 1].set_ylabel('Time (seconds)', fontweight='bold')
        axes[1, 1].set_title('Cumulative Training Time', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Training curves saved to {plot_path}")