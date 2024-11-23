from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from pill_dictionary import PillDictionary

app = Flask(__name__, static_url_path='/static')

# Define the CNN model
class PillCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(PillCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load the trained model
model = PillCNN(num_classes=9)
model.load_state_dict(torch.load('pill_classifier_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the pill dictionary
pill_dict = PillDictionary()

# Function to preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

# Function to make a prediction and display pill information
def predict_pill(image_path):
    image_tensor = preprocess_image(image_path)
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()

    # Display the image
    img = Image.open(image_path)
    img.show()

    # Display pill information
    predicted_pill = list(pill_dict.pill_info.keys())[predicted_class]
    info = pill_dict.get_pill_info(predicted_pill)
    return predicted_pill, info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/aboutus.html', methods=['GET'])
def about_us():
    return render_template('aboutus.html')

@app.route('/upload.html', methods=['GET'])
def upload():
    return render_template('upload.html', result=None)

@app.route('/identify', methods=['POST'])
def identify_pill():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # Save the file temporarily
        file_path = 'static/img/temp.jpg'
        file.save(file_path)

        # Example usage
        predicted_pill, info = predict_pill(file_path)

        # Return the prediction and information
        return render_template('result.html', predicted_pill=predicted_pill, info=info)

@app.route('/dictionary')
def pill_dictionary():
    # Example: Fetch information about a specific pill
    pill_name = request.args.get('pill_name')

    if pill_name:
        info = pill_dict.get_pill_info(pill_name)
        return render_template('dictionary.html', pill_name=pill_name, info=info)
    else:
        return "Pill name not provided"

if __name__ == '__main__':
    app.run(debug=True, port=1702)
