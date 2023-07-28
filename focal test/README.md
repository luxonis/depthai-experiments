# Utilities

This folder contains Focal test for cameras.

## Focal test

Firstly you need to install dependencies by running
``python3 install_requirements.py``
and then you can run focal test with two option; by displaying board or without it and cropping image.

## Croping taken image
The test will take as default that you want to just crop the taken frame into more small images and then check, how focused the images are. The parameters which determinates, how many crops do you have in images are ``-hs``=horizontal crops and ``-vs``=vertical crops.
The sample code, which you can run to get cropped images is:
```
python3 focal_test.py -hs 2 -vs 2 
```
to also display the cropped images, add ``-mts`` argument. To add the animations (not recommended for crop images) add ``-anim``.

## Taking the picture of board
The board, which is inside the folder, can be put anywhere in the room, because program will detect the charuco boards around it and crop the image so just the board will be shown. 
To use that feature, you need to add argumen ``-dbd``. In this option, argument ``-mts`` does not work, while argument ``-anim`` works.
Sample code would be
```
python3 focal_test.py -dbd -anim
```

