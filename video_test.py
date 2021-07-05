import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os

row, col = 224, 224
batch_size=32

train_dir='Input/train/'
val_dir='Input/val/'
model_path='models'

datagen = ImageDataGenerator()

train_dataset  =  datagen.flow_from_directory(directory = train_dir,
                                                   target_size = (row,col),
                                                   class_mode = 'categorical',
                                                   color_mode="rgb",
                                                   batch_size = batch_size)
valid_dataset = datagen.flow_from_directory(directory = val_dir,
                                                  target_size = (row,col),
                                                  class_mode = 'categorical',
                                                  color_mode="rgb",
                                                  batch_size = batch_size)

# load json and create model
json_file = open(os.path.join(model_path,'model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(os.path.join(model_path,'model.h5'))
print("Loaded model from disk")

model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

train_loss, train_accu = model.evaluate(train_dataset)
print("final train accuracy = {:.2f} ".format(train_accu*100))
print("final train loss = {:.2f} ".format(train_loss))

test_loss, test_accu = model.evaluate(valid_dataset)
print("validation accuracy = {:.2f}".format( test_accu*100))
print("validation loss = {:.2f}".format( test_loss))