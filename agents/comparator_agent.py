import torch  # PyTorch library for deep learning
import torch.nn as nn  # Neural network modules
from torchvision import transforms, models  # Preprocessing and pretrained vision models
from PIL import Image  # Python Imaging Library for image processing

class SiameseNetwork(nn.Module):
    """
    Siamese Network using a ResNet18 backbone to extract image embeddings.
    Compares two input images and outputs a similarity score between 0 and 1.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Load pretrained ResNet18 and remove its final classification layer
        base_cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base_cnn.fc = nn.Identity()  # Remove the last FC layer for feature extraction
        self.embedding = base_cnn  # Assign modified ResNet as embedding extractor

        # Fully connected layers to compute similarity from feature difference
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # Reduce dimensionality
            nn.ReLU(),            # Non-linearity
            nn.Linear(256, 1),    # Output single similarity score
            nn.Sigmoid()          # Normalize output to [0, 1]
        )

    def forward(self, x1, x2):
        """
        Forward pass of the Siamese Network.
        Args:
            x1: Tensor for the first image (shape: [1, 3, 224, 224])
            x2: Tensor for the second image (same shape)
        Returns:
            Tensor: similarity score between x1 and x2
        """
        f1 = self.embedding(x1)  # Extract features from image 1
        f2 = self.embedding(x2)  # Extract features from image 2
        dist = torch.abs(f1 - f2)  # Compute absolute feature difference
        out = self.fc(dist)  # Feed into fully connected network
        return out  # Return similarity score

class ComparatorAgent:
    """
    Comparator Agent that uses a trained Siamese Network to determine
    if an ATM image has been tampered with by comparing it to a reference.
    """
    def __init__(self, model_path="siamese_atm_model.pth"):
        """
        Initialize the agent with the trained model and preprocessing pipeline.
        Args:
            model_path (str): Path to the saved Siamese model weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.model = SiameseNetwork().to(self.device)  # Load model to device
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))  # Load weights
        self.model.eval()  # Set model to evaluation mode

        # Preprocessing: resize and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def compare(self, img1_input, img2_input, threshold=0.5):
        """
        Compares two ATM images and returns similarity score and classification.
        Args:
            img1_input: Path or PIL image of the reference ATM image
            img2_input: Path or PIL image of the suspect ATM image
            threshold (float): Similarity threshold to determine tampering
        Returns:
            tuple: (similarity_score (float), "Tampered"/"Clean")
        """
        # Helper function to load and transform input
        def load(image_input):
            if isinstance(image_input, str):
                # If input is a file path, open and preprocess
                return self.transform(Image.open(image_input).convert("RGB")).unsqueeze(0).to(self.device)
            elif isinstance(image_input, Image.Image):
                # If input is a PIL image, just preprocess
                return self.transform(image_input.convert("RGB")).unsqueeze(0).to(self.device)
            else:
                raise ValueError("Unsupported input: must be file path or PIL.Image")

        img1 = load(img1_input)  # Load and preprocess clean image
        img2 = load(img2_input)  # Load and preprocess suspect image

        # Inference without gradient computation
        with torch.no_grad():
            score = self.model(img1, img2).item()  # Get similarity score

        # Classify based on threshold
        result = "Tampered" if score >= threshold else "Clean"
        return score, result  # Return similarity score and result