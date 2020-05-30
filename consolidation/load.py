import numpy as np
import keras.models
import tensorflow as tf
from keras.models import model_from_json
from pickle import load

def init(): 
	json_file = open('model\model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = tf.keras.models.model_from_json(loaded_model_json)
	loaded_model.load_weights("model\model.h5")
	print("Awesome, your model has been loaded from disk! Cool beans!")

	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	return loaded_model