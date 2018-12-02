# Localization
This repository performs localization using a custom-trained version of [YOLO](https://pjreddie.com/darknet/yolo/) on the [MIO-TCD](http://podoce.dinf.usherbrooke.ca/challenge/dataset/) dataset.

## Prequisites
Requires Nvidia drivers, CUDA, cuDNN for training. Use [Anaconda](https://www.anaconda.com/download/#linux) to install other common librairies and OpenCV.

## Training
- Clone [darknet](https://github.com/AlexeyAB/darknet) into the `yolo` directory.
- Compile using these [instructions](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux).
- Place dataset under yolo/data, then run `setup.py` to generate labels and text files required for YOLO.
- Copy `data`, `localization.data` and `yolo-localization.cfg` into `darknet` directory.
- Download pre-trained [weights](https://pjreddie.com/media/files/darknet53.conv.74) for convolutional layers into `darknet` directory.
- From `darknet` directory, run `./darknet detector train localization.data yolo-localization.cfg darknet53.conv.74`.
- Weights will be stored under `darknet/backup` every 100 iterations.

## Evaluation
Import model using your favorite dl library or use `Localizer.ipynb`.