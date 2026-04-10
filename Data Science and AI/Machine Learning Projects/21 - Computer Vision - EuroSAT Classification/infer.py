import os, glob, csv, argparse, numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from train import resnet18_cifar

def load_model(out_dir="outputs", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(out_dir,"best.pt"), map_location=device)
    classes = ckpt["classes"]
    model = resnet18_cifar(num_classes=len(classes))
    model.load_state_dict(ckpt["model"]); model.to(device).eval()
    return model, classes, device

@torch.no_grad()
def predict_tensor(model, x, device):
    x = x.to(device)
    logits = model(x.unsqueeze(0))
    pred = logits.argmax(1).item()
    return pred

def load_image_64(path):
    img = Image.open(path).convert("RGB").resize((64,64))
    tf = transforms.ToTensor()
    return tf(img)

def infer_test_split(data_root="EuroSAT", out_dir="outputs"):
    """Run inference over the saved 10% TEST split and save CSV."""
    model, classes, device = load_model(out_dir)
    splits = np.load(os.path.join(out_dir,"splits_80_10_10.npz"), allow_pickle=True)
    test_idx = splits["test"]

    ds = datasets.ImageFolder(data_root, transform=transforms.ToTensor())
    test = Subset(ds, test_idx)

    rows = [("filepath","true_class","pred_class")]
    ok = 0
    for i in range(len(test)):
        (x,y) = test[i]
        # Recover original path from the underlying dataset
        path, true_idx = ds.samples[test_idx[i]]
        pred_idx = predict_tensor(model, x, device)
        rows.append((path, classes[true_idx], classes[pred_idx]))
        if pred_idx == true_idx:
            ok += 1

    acc = 100.0 * ok / len(test)
    print(f"[Inference on TEST split] images={len(test)}  accuracy={acc:.2f}%")
    csv_path = os.path.join(out_dir, "preds_test.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerows(rows)
    print(f"Saved test predictions → {csv_path}")

def infer_custom_glob(images_glob, out_dir="outputs"):
    """Run inference over any images you specify via glob and save CSV."""
    model, classes, device = load_model(out_dir)
    paths = sorted(glob.glob(images_glob))
    if not paths:
        print(f"No files matched: {images_glob}")
        return
    rows = [("filepath","pred_class")]
    for p in paths:
        x = load_image_64(p)
        pred_idx = predict_tensor(model, x, device)
        rows.append((p, classes[pred_idx]))
        print(f"{p} → {classes[pred_idx]}")
    csv_path = os.path.join(out_dir, "preds_custom.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerows(rows)
    print(f"Saved custom predictions → {csv_path}")

if _name_ == "_main_":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_glob", type=str, default=None,
                    help='Optional glob for custom images, e.g. "test_images/*.jpg"')
    ap.add_argument("--data_root", type=str, default="EuroSAT")
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    if args.images_glob:
        infer_custom_glob(args.images_glob, out_dir=args.out_dir)
    else:
        infer_test_split(data_root=args.data_root, out_dir=args.out_dir)