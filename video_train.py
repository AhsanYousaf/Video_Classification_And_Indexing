import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten,Dropout,BatchNormalization,Activation
from keras.models import Model,Sequential
# from keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import regularizers
import datetime
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


row, col = 224, 224
batch_size=32
epochs= 200

train_dir='Input/train/'
val_dir='Input/val/'


train_datagen = ImageDataGenerator(rescale = 1./255,rotation_range=5, zoom_range=0.1)
valid_datagen = ImageDataGenerator(rescale = 1./255)

train_dataset  = train_datagen.flow_from_directory(directory = train_dir,
                                                   target_size = (row,col),
                                                   class_mode = 'categorical',
                                                   color_mode="rgb",
                                                   batch_size = batch_size)
valid_dataset = valid_datagen.flow_from_directory(directory = val_dir,
                                                  target_size = (row,col),
                                                  class_mode = 'categorical',
                                                  color_mode="rgb",
                                                  batch_size = batch_size)

# Classes
print(train_dataset.class_indices)


base_model = tf.keras.applications.ResNet50(input_shape=(row ,col,3),include_top=True,weights="imagenet")

for layer in base_model.layers[:-4]:
    layer.trainable=True
    
model=tf.keras.models.Sequential() 
model.add(base_model)
model.add(Dropout(0.2))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(32,kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64,kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1024,kernel_initializer='he_uniform',))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(8,activation='softmax'))
model.summary()


model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


tf.keras.utils.plot_model(model,to_file='models/model.png', show_shapes=True, show_layer_names=True)

chk_path='models/model.h5'
log_dir = "models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint = ModelCheckpoint(filepath=chk_path,
                             save_best_only=True,
                             verbose=1,
                             mode='auto',
                             moniter='val_loss',
                             save_weights_only=True)

earlystop = EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=8, 
                          verbose=1, 
                          restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=4, 
                              verbose=1, 
                              min_delta=0.0001)
csv_logger = CSVLogger('models/model_training_logs.log')

callbacks = [ reduce_lr, csv_logger,earlystop,checkpoint]

trained_model= model.fit(train_dataset,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=valid_dataset,
                                    callbacks=callbacks
                                    )

# serialize model to JSON
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/later_model.h5")
print("Saved model to disk")


plt.figure(figsize=(14,5))
plt.subplot(1,2,2)
plt.plot(trained_model.history['accuracy'])
plt.plot(trained_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'valid'], loc='upper left')

plt.subplot(1,2,1)
plt.plot(trained_model.history['loss'])
plt.plot(trained_model.history['val_loss'])
plt.title('model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('train_results/training_results.png')
plt.show()