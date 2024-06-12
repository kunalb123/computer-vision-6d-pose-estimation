This repo contains code to find the 6D position and orientation of an object in a 2D image.

We utilize the LineMOD Dataset and the test and train data can be accessed via this link:
https://bop.felk.cvut.cz/datasets/

Our makes a few assumptions about the structure of your directories.
First, when downloading the data from the website place the following folders in the root of the repo:
lm
lm_models
test
train

Note: we do not use train_pbr we specifically use the training folder that contains 15 sub-folders representing the 15 objects. In the train folders these images should be rendered. We also use the test folder that also contains 15 sub-folders representing 15 obejcts. In the test folder, the images of the objects are real. The naming is a bit confusing, but we actually combined all images for an object so that our set of samples includes rendered and real images, then do an 80-20 split for training and testing.

Next, create a folder at the root of the directory called `model_checkpoints/`. This will store the checkpoints after each epoch that the model trains at a given configuration.

Finally, there must be an `annotations/` file directory that contains annotation files for the data and handles the conversion of BOP (the way the data is formatted when downloaded) to COCO (the way we need the data to be formatted for our system)