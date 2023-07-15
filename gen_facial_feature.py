from tensorflow.keras.models import model_from_json
from autoencoder import ConvAutoencoder
import matplotlib.pyplot as plt
import numpy as np
import cv2
from flask import Flask, request
from flask_cors import CORS, cross_origin
import base64
import json

app = Flask(__name__)
cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
autoencoder = model_from_json(loaded_model_json)
# load weights into new model
autoencoder.load_weights("model.h5")


# load json and create model encoder
json_file = open('model_encoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder = model_from_json(loaded_model_json)
# load weights into new model
encoder.load_weights("model_encoder.h5")


# load json and create model decoder
json_file = open('model_decoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
decoder = model_from_json(loaded_model_json)
# load weights into new model
decoder.load_weights("model_decoder.h5")

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

@app.route('/api/facevector', methods=['GET'])
@cross_origin()
def gen_feature():
    img_str = request.json['img']
    img = readb64(img_str)
    img_expandim = np.expand_dims(img, axis=0)
    encoded = np.array(encoder(img_expandim))
    print(encoded.shape)
    facevec = json.dumps({'vector':encoded[0].tolist()})
    return facevec
    

if __name__ == "__main__":
    app.run(host="0.0.0.0")


# img = cv2.imread('C:\\Users\\joe\\Downloads\\faceage\\20-50\\20-50\\test\\20\\28521.jpg').astype("float32") / 255.0
# img1 = np.expand_dims(img, axis=0)

# img_encoded = autoencoder.predict(img1, verbose=0)
# res =  (img_encoded * 255).astype("uint8")
# cv2.imshow('result',res[0])
# cv2.waitKey(0)


# encoded = encoder(img1)
# print(encoded[0])


# # decoded_en = np.array(decoder(encoded))
# # outimg = (decoded_en[1]*225).astype("uint8")
# # cv2.imshow("encoded", outimg)
# # cv2.waitKey(0)

