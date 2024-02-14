import argparse
import torch
import json
from PIL import Image
from model_utils import load_checkpoint, process_image, predict

def parse_args_predict():
    parser = argparse.ArgumentParser(description="Predict image class using a trained deep learning model.")
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('checkpoint', help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=1, help='Top K most likely classes (default: 1)')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names JSON file (default: cat_to_name.json)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    
    return parser.parse_args()

def main_predict():
    args = parse_args_predict()
    
    # Check if GPU is available and the user wants to use it
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load the checkpoint
    model, class_to_idx = load_checkpoint(args.checkpoint)
    
    # Process the input image
    image = process_image(args.image_path)
    
    # Move model and image to the selected device (GPU or CPU)
    model.to(device)
    image = image.to(device)
    
    # Perform prediction with top K classes
    top_classes, top_probabilities = predict(model, image, args.top_k, device)
    
    # Load category names mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Display category names in the output
    print(f"Image Categories:")
    for i in range(len(top_classes)):
        class_name = cat_to_name[top_classes[i]]
        print(f"Top {i+1} - Category: {class_name}, Probability: {top_probabilities[i]}")

if __name__ == "__main__":
    main_predict()
