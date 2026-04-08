"""
Zero-shot image classification using CLIP.
Usage: python clip_zeroshot.py --image your_image.jpg --categories cat dog car bicycle person
"""

import argparse
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def classify(image_path, categories, model_name="ViT-B/32", top_k=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    print(f"Loaded model: {model_name}")

    # Load and preprocess image
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    # Tokenize text descriptions
    text_templates = [f"a photo of a {c}" for c in categories]
    text_tokens = clip.tokenize(text_templates).to(device)

    # Forward pass
    with torch.no_grad():
        image_features = model.encode_image(image).float()
        text_features = model.encode_text(text_tokens).float()

        # L2 normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity -> softmax probabilities
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs, labels = logits.cpu().topk(top_k, dim=-1)

    # Print results
    print(f"\nResults for: {image_path}")
    print("-" * 50)
    for rank in range(top_k):
        cat = categories[labels[0, rank]]
        prob = probs[0, rank].item()
        print(f"  Top-{rank+1}: {cat:20s}  {prob:.2%}")

    # Plot
    _plot(image_path, categories, probs[0], labels[0], top_k)


def _plot(image_path, categories, probs, labels, top_k):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Original image
    img = Image.open(image_path).convert("RGB")
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Input Image")

    # Top-k bar chart
    y = np.arange(top_k)
    cats = [categories[labels[i].item()] for i in range(top_k)]
    bars = ax1.barh(y, probs.numpy(), color="steelblue")
    ax1.set_yticks(y)
    ax1.set_yticklabels(cats)
    ax1.set_xlabel("Probability")
    ax1.invert_yaxis()

    # Similarity heatmap for all categories
    im = ax2.barh(categories, probs.numpy(), color="steelblue")
    ax2.set_xlabel("Probability")
    ax2.set_title(f"Top {top_k} of {len(categories)} categories")

    # Annotate top-k bars
    for i, (bar, val) in enumerate(zip(bars, probs.numpy())):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.2%}", va="center")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP zero-shot image classification")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--categories", nargs="+", default=["cat", "dog", "car", "bicycle", "person"],
                        help="Candidate categories")
    parser.add_argument("--model", type=str, default="ViT-B/32",
                        choices=clip.available_models(), help="CLIP model to use")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to show")
    args = parser.parse_args()

    classify(args.image, args.categories, args.model, args.top_k)
