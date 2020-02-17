import keras
from keras import models
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import glob
import sys 

img_size = 64 # set universal size for images -- images will also be grayscale
batch_size = 64

def trueSampleName (sample):
	i = int(sample.replace("Sample", ""))
	name = ""
	if i <= 10:
		name = str(i-1)
	elif i <= 36:
		name = chr(i-11+65)
	else:
		name = chr(i-37+97)

	return name

# load a saved gender detection model
def load_model ():
	model = make_model()
	model.load_weights("model.h5")
	return model

# create the CNN to detect gender
def make_model ():

	num_classes = len(glob.glob('data/train_data/Sample*'))

	model = Sequential()
	model.add(Conv2D(48, kernel_size=5, strides=(2,2),input_shape=(img_size, img_size, 1), activation="relu", padding='same'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(MaxPooling2D())
	model.add(BatchNormalization())

	model.add(Conv2D(48, kernel_size=5, strides=(2,2), activation="relu", padding='same'))
	model.add(MaxPooling2D())

	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(BatchNormalization())

	#model.add(Conv2D(64, kernel_size=3, activation="relu", padding='same'))
	#model.add(MaxPooling2D())

	model.add(Conv2D(32, kernel_size=3, activation="relu", padding='same'))
	model.add(MaxPooling2D())
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(BatchNormalization())
	model.add(Dropout(0.3))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.4))

	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.4))

	model.add(Dense(num_classes, activation='softmax'))#, activity_regularizer=regularizers.l1(0.01)))

	model.compile(
		loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy']
    )
	return model

# train a gender detection CNN
def train_model (model, epochs):

	# we create an image data generator that randomly modifies an image
	# so that we have better & more noisy data to train on
	# we will get better generalisation performance
	train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=False)

	# generator will read pictures found in the specified directory
	# classes are detected based on the folder
	# wil create random batches of the data it finds
	train_generator = train_datagen.flow_from_directory(
        'data/train_data',  # this is the target directory
        target_size=(img_size, img_size),  # all images will be resized to 150x150
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

	# callbacks=[EarlyStopping(patience=3, restore_best_weights=True), 
    # ReduceLROnPlateau(patience=2), 
    # ModelCheckpoint(filepath='gender_model_chk.h5', save_best_only=True)] 
	# finally, we train the model
	model.fit_generator(
        train_generator,
        steps_per_epoch=4096 // batch_size,
        epochs=epochs
        #callbacks=callbacks,
       # validation_steps=500 // batch_size
       )

	
	# save the model
	model.save_weights('model.h5')  # always save your weights after training or during training
	return model

def show_confusion_matrix (model):
		# this is a similar generator, for validation data
	validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'data/test_data',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False)


	y_pred = model.predict_generator(validation_generator)
	y_pred = np.argmax(y_pred, axis=1)

	classes = validation_generator.classes

	accuracy = [1 if e else 0 for e in y_pred-classes == 0]
	accuracy = sum(accuracy)/len(y_pred)
	print ("testing accuracy: " + str( accuracy ))

	class_weights = class_weight.compute_class_weight('balanced', np.unique(classes), classes)

	cm = confusion_matrix(validation_generator.classes, y_pred)
	cm = [ [ round( float(cm[i][j]) * class_weights[i], 2) for j in range(0, len(cm[i]))] for i in range(0, len(cm))  ]

	class_names = validation_generator.class_indices
	class_values = [ trueSampleName(k) for k in class_names ]
	class_indices = [ v for v in class_names.values() ]

	plt.imshow(cm, cmap=plt.cm.Blues)
	plt.xlabel("Predicted labels")
	plt.ylabel("True labels")
	plt.xticks(class_indices, class_values)
	plt.yticks(class_indices, (class_values))
	plt.title('Confusion Matrix (Testing Accuracy ' + str(int(accuracy*100)) + '%)')
	plt.colorbar()
	plt.show()

def show_intermediate_images (model, layers):
	layer_outputs = [model.layers[layer].output for layer in layers]
	activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

	testImageNames = ["data/train_data/Sample011/img011-00003.png"]
	images = []

	for imageName in testImageNames:
		img = image.load_img(imageName, target_size=(img_size, img_size), color_mode='grayscale')
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)

		images.append( x )
	images = np.vstack(images) 

	activations = activation_model.predict(images)

	layer_names = [model.layers[layer].name for layer in layers]

	images_per_row = 8

	for layer_name, layer_activation in zip(layer_names, activations):
		n_features = layer_activation.shape[-1]
		size = layer_activation.shape[1]
		n_cols = n_features // images_per_row
		display_grid = np.zeros((size * n_cols, images_per_row * size))

		for col in range(n_cols):
			for row in range(images_per_row):
				channel_image = layer_activation[0, :, :, col * images_per_row + row]
				channel_image -= channel_image.mean()
				channel_image /= channel_image.std()
				channel_image *= 64
				channel_image += 128
				channel_image = np.clip(channel_image, 0, 255).astype('uint8')
				display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
		scale = 1. / size

		plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
		plt.title(layer_name)
		plt.grid(False)

		plt.imshow(display_grid, aspect='auto')
	plt.show()

arguments = sys.argv
del arguments[0]

if len(arguments) <= 0:
	print(
		"need more arguments\n"
		+ "-load to load the stored model\n"
		+ "-new to create a new model\n"
		+ "-train N to train the model with N epochs\n"
		+ "-confmatrix to show the confusion matrix\n"
	)
	exit(-1)

if arguments[0] == "-load":
	model = load_model()
	del arguments[0]
else:
	model = make_model()
	if arguments[0] == "-new":
		del arguments[0]

if len(arguments) > 0 and arguments[0] == "-train":
	del arguments[0]

	try:
		epochs = int(arguments[0])
		del arguments[0]
	except:
		epochs = 10

	model = train_model(model, epochs)

if len(arguments) > 0 and arguments[0] == "-predict":
	del arguments[0]

	filename = arguments[0]
	img = image.load_img(filename, target_size=(img_size, img_size), color_mode='grayscale')
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)

	classes = model.predict_classes(img, batch_size=batch_size)
	print ("image at " + filename + " is of character: " +  trueSampleName( str(classes[0]+1) ))

if len(arguments) > 0:
	if arguments[0] == "-confmatrix":
		del arguments[0]
		show_confusion_matrix(model)
	elif arguments[0] == "-intermediates":
		del arguments[0]
		show_intermediate_images(model, [0, 4])