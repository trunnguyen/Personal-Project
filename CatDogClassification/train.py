import torch
from model import DogCatCNN
from data import get_dataloaders
from trainer import train, evaluate, plot_history, visualize_predictions
from log import get_logger, save_history

def main():
    logger = get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = DogCatCNN().to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")

    train_loader, val_loader = get_dataloaders(
        train_dir='./data/train',
        valid_dir='./data/valid',
        batch_size=64,
    )
    class_names = train_loader.dataset.classes
    logger.info(f"Classes: {class_names}")

    history = train(
        model, train_loader, val_loader, device,
        epochs=20, patience=5, checkpoint_path='best_model.pth',
        class_names=class_names
    )

    history_path = save_history(history, log_dir='logs', name='history')
    logger.info(f"History saved to {history_path}")

    plot_history(history)

    final_loss, final_acc = evaluate(model, val_loader, device)
    logger.info(f"Final loss (best model: {final_loss:.4f}, final accuracy: {final_acc:.2f}%")

    visualize_predictions(model, val_loader, device ,class_names=class_names)

if __name__ == '__main__':
    main()