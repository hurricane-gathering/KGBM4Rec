from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import random

app = Flask(__name__)

model = load_model(r'C:\myfiles\files\projects\python\results\tf100.keras')

input_shape = (512, 512, 3)

labels = ["卫衣",
    "小羽绒服",
    "大羽绒服",
    "大白褂",
    "小棉袄",
    "大棉袄",
    "冲锋衣",
    "保暖衣",
    "长款羽绒服",
    "T恤"]

def preprocess_image(image):
    img = Image.open(image)
                              
    img = img.resize((input_shape[0], input_shape[1]))
    
    img_array = np.array(img) / 255.0
    return img_array

@app.route('/classify', methods=['POST'])
def classify():
    image = request.files['image']
    
    preprocessed_image = preprocess_image(image)
    
    img_array = np.reshape(preprocessed_image, (1, *input_shape))
    
    predictions = model.predict(img_array)
    
    top3_indices = np.argsort(predictions)[0, -3:][::-1]  
    top3_labels = [labels[idx] for idx in top3_indices]
    top3_probabilities = [float(predictions[0, idx]) for idx in top3_indices]
    
    result = {
        'predictions': [{
            'label': label,
            'probability': prob
        } for label, prob in zip(top3_labels, top3_probabilities)]
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
