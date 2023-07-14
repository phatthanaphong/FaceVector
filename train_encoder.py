
from tensorflow.keras.optimizers import Adam
from autoencoder import ConvAutoencoder
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

EPOCHS = 100
BS = 32

path = 'C:\\Users\\joe\\Downloads\\faceage\\20-50\\20-50\\train\\'

train_x = []
test_x = []

no = 5000
i = 0
for folder in os.listdir(path):
    for filename in os.listdir(os.path.join(path,folder)):
        img = cv2.imread(path+folder+'//'+filename, 1)
        if i < 4500:
            train_x.append(img)
        else:
            test_x.append(img)
        i+=1
        if(i==no):
            break
    if(i==no):
        break

print(len(train_x))

trainX = np.array(train_x)
testX = np.array(test_x)

trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoencoder.build(128, 128, 3)
opt = Adam(lr=1e-3)
autoencoder.compile(loss="mse", optimizer=opt)
# train the convolutional autoencoder
H = autoencoder.fit(
	trainX, trainX,
	validation_data=(testX, testX),
	epochs=EPOCHS,
	batch_size=BS)


#save the autoencoder model 
# serialize model to JSON
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")


# serialize encoder model to JSON
model_encoder_json = encoder.to_json()
with open("model_encoder.json", "w") as json_file:
    json_file.write(model_encoder_json)
# serialize weights to HDF5
encoder.save_weights("model_encoder.h5")
print("Saved encoder model to disk")



# serialize decoder model to JSON
model_decoder_json = decoder.to_json()
with open("model_decoder.json", "w") as json_file:
    json_file.write(model_decoder_json)
# serialize weights to HDF5
decoder.save_weights("model_decoder.h5")
print("Saved decoder model to disk")


# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('loss.png')
# plt.show()