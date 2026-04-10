import os, numpy as np, torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from train import resnet18_cifar

# optional: detailed metrics
from sklearn.metrics import classification_report, confusion_matrix

@torch.no_grad()
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join("outputs","best.pt"), map_location=dev)
    classes = ckpt["classes"]

    # load model
    model = resnet18_cifar(num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()

    # load saved test indices
    splits = np.load(os.path.join("outputs","splits_80_10_10.npz"), allow_pickle=True)
    test_idx = splits["test"]

    # dataset & loader
    tf = transforms.ToTensor()
    ds = datasets.ImageFolder("EuroSAT", transform=tf)   # same root you used for training
    test = Subset(ds, test_idx)
    dl = DataLoader(test, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    total, correct = 0, 0
    all_y, all_p = [], []
    for x,y in dl:
        x,y = x.to(dev), y.to(dev)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_y += y.cpu().tolist()
        all_p += pred.cpu().tolist()

    acc = 100 * correct / total
    print(f"Test accuracy (10% hold-out): {acc:.2f}%")

    # detailed per-class report
    print("\nClassification report:")
    print(classification_report(all_y, all_p, target_names=classes, digits=4))

    # confusion matrix (optional text print)
    cm = confusion_matrix(all_y, all_p, labels=list(range(len(classes))))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

if _name_ == "_main_":
    main()