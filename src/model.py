import io
import os.path

import torch
from PIL import Image
from torchvision import models, transforms

from src.utils import colored_print

RESULTS_DIR = 'result'

flower_classes = {0: 'ромашка', 1: 'одуванчик', 2: 'лаванда', 3: 'лилия', 4: 'лотос', 5: 'орхидея', 6: 'роза',
                  7: 'подсолнух', 8: 'тюльпан'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_name):
    new_model = models.resnet50(pretrained=False)

    number_of_classes, number_of_features = len(flower_classes), new_model.fc.in_features

    new_model.fc = torch.nn.Linear(number_of_features, number_of_classes)

    new_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f'{model_name}.pth'), map_location=device))

    new_model.to(device)
    new_model.eval()

    return new_model


model = load_model('resnet50')
colored_print("Loaded model...", 'g')


def predict(image):
    image = Image.open(io.BytesIO(image))
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    return predicted.item(), confidence.item()


def predict_flower(image):
    flower_key, confidence = predict(image)

    return flower_classes[flower_key], confidence
