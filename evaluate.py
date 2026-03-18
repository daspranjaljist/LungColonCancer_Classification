import torch
import numpy as np
from sklearn.metrics import classification_report
from models.dual_net import DualTransferNet
from utils.dataset import get_data_loaders
# Import the custom plotting functions we created
from utils.metrics import plot_confusion_matrix, plot_roc_curve

def main():
    # -------------------------
    # 1. Configuration
    # -------------------------
    DATA_DIR = './data/lung_colon_image_set/'
    MODEL_PATH = 'best_model.pth'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")

    # -------------------------
    # 2. Load Data & Model
    # -------------------------
    print("Loading dataset...")
    _, _, test_loader, class_names = get_data_loaders(DATA_DIR, batch_size=32)
    print(f"Classes found: {class_names}")
    
    model = DualTransferNet(num_classes=len(class_names)).to(DEVICE)
    
    print("Loading trained model...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first using train.py")
        return
        
    model.eval()

    # -------------------------
    # 3. Run Evaluation
    # -------------------------
    y_true = []
    y_pred = []
    y_score = [] # To store probabilities for ROC curve

    print("Evaluating on Test Set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            
            outputs = model(inputs)
            
            # Get probabilities (Softmax) for ROC curve
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predicted class index
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # -------------------------
    # 4. Print Classification Report
    # -------------------------
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # -------------------------
    # 5. Plot Confusion Matrix
    # -------------------------
    print("\nGenerating Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png')

    # -------------------------
    # 6. Plot ROC Curve
    # -------------------------
    print("\nGenerating ROC Curve...")
    plot_roc_curve(y_true, y_score, class_names, save_path='results/roc_curve.png')
    
    print("\nEvaluation Complete. Results saved in 'results/' folder.")

if __name__ == "__main__":
    main()