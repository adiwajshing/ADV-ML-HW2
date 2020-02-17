import os
import glob
import random
from shutil import copyfile

def load_data (directory, seed):
	random.seed(seed)

	good_img_directory = "Img/GoodImg/Bmp/"
	bad_img_directory = "Img/BadImag/Bmp/"

	samples = glob.glob(good_img_directory + "*")
	samples.sort()

	for sample in samples:
		folder = sample.replace(good_img_directory, "")

		goodImgs = glob.glob(good_img_directory + folder + "/*.png")
		badImgs = glob.glob(bad_img_directory + folder + "/*.png")

		imgs = set(goodImgs+badImgs)

		test_imgs = set(random.sample(imgs, int(len(imgs)*0.2)))
		training_imgs = imgs-test_imgs

		test_directory = directory + "/test_data/" + "/" + folder + "/"
		train_directory = directory + "/train_data" + "/" + folder + "/"

		os.makedirs(test_directory, exist_ok=True)
		os.makedirs(train_directory, exist_ok=True)

		for filename in training_imgs:
			lname = filename.split("/")[-1]
			copyfile(filename, train_directory + lname)
		for filename in test_imgs:
			lname = filename.split("/")[-1]
			copyfile(filename, test_directory + lname)

load_data("data", random.randint(0, 1000000))