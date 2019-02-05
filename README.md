# DeepSeg - CNN n°1

CNN n°1 is an encoder-decoder neural network based on an U-Net architecture.

## Dependency

Following modules are used in the project:

    * python >= 3.6
    * PIL >= 5.2.0
    * torch==1.0.0
    * torchvision==0.2.1
    * pybullet==2.4.2
    * numpy==1.16.0
    * opencv-python==4.0.0.21
    * moderngl==5.5.0

## Dataset generation

Dataset generation is made via main.py file.
* ```-p``` to activate random number of pieces mode (between 10 and 50)
* ```-l``` to activate random luminosity mode  

```
python3 main.py -l -p
```

Data is then stored in ```input/background``` for the the images and ```input/truth``` for the images masks.

## Training

Once dataset is generated, the network has to be trained by using one of the following commands.

```
python3 cnn1.py -t
python3 cnn1.py --train
```

By default GPU will be used if available, otherwise CPU will be used (not recommanded).

At the end of the training, the model is saved into ```cnn1.pth``` so it can be loaded later.

## Evaluate / Predict

Once the network is trained, we can predict masks from the test set located in ```/input/to_predict``` and corresponding masks from ```/input/to_predict_truth``` are used to compute statistics (loss).

To run the program in predict mode use one of the following commands. Note that this is the default mode.

```
python3 cnn1.py
python3 cnn1.py -p
python3 cnn1.py --predict
```

## Tests

You can see graphical results by launching the ```cnn1.ipynb``` notebook.

## Authors

* **Antoine Marjault** - [GIT](https://gitlab.univ-nantes.fr/E177646T)
* **Sébastien Berlioux** - [GIT](https://gitlab.univ-nantes.fr/E177663M)
* **Damien Farce** - [GIT](https://gitlab.univ-nantes.fr/E146084M)
* **Julien Langlois** - *Initial work on data generation* - [GIT](https://gitlab.univ-nantes.fr/E15H781N)
