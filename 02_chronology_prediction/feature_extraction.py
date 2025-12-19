import os
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2
from skimage.feature import hog
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


from utils import get_device

PATH = "../../data/"
PATH_IMAGES= PATH + "images/"
DEVICE = get_device(verbose=False)

def missing_image(image_filename):
    return not isinstance(image_filename, str) or len(image_filename) == 0

# TF-IDF

def get_tfidf_vectorizer(max_features=300):
    vectorizer = TfidfVectorizer(
        max_features=max_features,  # Use top N important words
        stop_words='english'
    )
    return vectorizer


# BERT

BERT_MODEL_NAME = "bert-base-uncased"  # "sentence-transformers/all-MiniLM-L6-v2" for optimized embeddings
BERT_TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
BERT_MODEL = AutoModel.from_pretrained(BERT_MODEL_NAME)

def get_bert_embedding(text):

    if pd.isna(text) or text.strip() == "":
        return torch.zeros(768).to(DEVICE)  # Fallback for empty text

    inputs = BERT_TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Move each tensor in the dict to CUDA
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
    BERT_MODEL.to(DEVICE)  # Make sure the model is on CUDA too!

    with torch.no_grad():
        outputs = BERT_MODEL(**inputs)

    bert_embedding_tensor = outputs.last_hidden_state[:, 0, :].squeeze()
    return bert_embedding_tensor


# CANNY - Edge Detection

def cv_read(image_filename):
    return cv2.imread(
        os.path.join(PATH_IMAGES, image_filename),
        cv2.IMREAD_GRAYSCALE
    )

def extract_canny_features(image_filename):
    if missing_image(image_filename):
        return 0.0

    image = cv_read(image_filename)
    image = cv2.resize(image, (128, 128))
    edges = cv2.Canny(image, threshold1=100, threshold2=200)
    canny_sum = np.sum(edges) / 255.0  # Edge density

    return canny_sum


# HOG - Histogram of Oriented Gradients

RESIZE_DIM = (128, 128)
ORIENTATIONS = 9
PIXELS_PER_CELL = (12, 12)
CELLS_PER_BLOCK = (2, 2)

HOG_LEN = ((RESIZE_DIM[0] // PIXELS_PER_CELL[0] - 1) *
           (RESIZE_DIM[1] // PIXELS_PER_CELL[1] - 1) *
           CELLS_PER_BLOCK[0] * CELLS_PER_BLOCK[1] *
           ORIENTATIONS)

def extract_hog_features(image_filename, return_image=False):
    if missing_image(image_filename):
        return np.zeros(HOG_LEN).tolist()

    image = cv_read(image_filename)
    image = cv2.resize(image, RESIZE_DIM)
    hog_vec, hog_image = hog(
        image,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )

    return (hog_vec, hog_image) if return_image else hog_vec.tolist()


# RESNET

RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

RESNET_MODEL = resnet50(weights='DEFAULT').to(device=DEVICE)
RESNET_MODEL = torch.nn.Sequential(*list(RESNET_MODEL.children())[:-1])

def extract_resnet_features(image_filename):
    if missing_image(image_filename):
        return torch.zeros(2048).to(DEVICE)  # if missing image return zero tensor

    image = Image.open(PATH_IMAGES + image_filename).convert("RGB")
    image_tensor = RESNET_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = RESNET_MODEL(image_tensor).squeeze()
    return features

# ViT

VIT_EXTRACTOR = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
VIT_MODEL = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').eval().cuda()

def extract_vit_features(image_filename):
    if missing_image(image_filename):
        return torch.zeros(768).to("cuda")  # Placeholder for missing images

    image = Image.open(PATH_IMAGES + image_filename).convert("RGB")
    inputs = VIT_EXTRACTOR(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = VIT_MODEL(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()