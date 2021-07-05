import os
import tensorflow as tf
import cv2
import numpy as np


model_path='models'
source='C:/Users/rehan/Downloads/Video/9convert.com - Prichard Colon VS Terrel Williams.mp4'
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

classes={0:'basketball',1: 'boxing', 
         2:'cricket',3: 'formula1', 
         4:'kabaddi', 5:'swimming', 
         6:'table_tennis',7: 'weight_lifting'}

cap = cv2.VideoCapture(source)
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
      # print(frame.shape)
      pred_img = cv2.resize(frame,(224,224))
      pred_img=np.expand_dims(pred_img, axis=0)
      # print(pred_img.shape)
      prediction = model.predict(pred_img)
      maxindex = int(np.argmax(prediction))
      sport=classes[maxindex]
      print("Sport is ",sport)
      image = cv2.putText(frame, sport, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
      cv2.imshow('Predicted Sport',image)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    