from flask import Flask, request
from flask_cors import CORS
import torch
import cv2
import numpy as np
import base64

import model as mdl

PATH = './steak_net.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mdl.ResNet(mdl.ResidualBlock, [3, 4, 6, 3], 6).to(device)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

class_dict = {0:'well done', 1:'medium well done', 2:'medium', 3 : 'medium rare', 4: 'rare', 5: 'blue rare'}

def findDoneness(img):
    x = cv2.imdecode(img, cv2.IMREAD_COLOR).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    x = cv2.resize(x, (416, 416))
    # x = self.transform(x)
    x = np.rollaxis(x, 2)
    x = torch.as_tensor(x, device=device)
    x = x.float().unsqueeze(0)
    outputClass, boundingbox = model(x)
    _, predicted = torch.max(outputClass.data, 1)
    print(predicted.item())
    return class_dict[predicted.item()]


app = Flask(__name__)
CORS(app)

@app.route('/identify', methods=["POST"])
def identify():
    img = request.files['image']
    img = img.read()
    img = np.fromstring(img, np.uint8)
    return {'Doneness' :findDoneness(img)}, 200

@app.route('/identifyJson', methods=["POST"])
def identifyJson():
    img = request.get_json()
    img = img.get('dataUrl')
    img = img.split(",")[-1]
    img = base64.b64decode(img)
    img = np.fromstring(img, np.uint8)
    return {'Doneness' :findDoneness(img)}, 200