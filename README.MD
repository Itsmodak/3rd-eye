In this repos we build our own vehicle detector or any other object detector from different classes using different rcnn models with tensorflow

**Note:** I am not provide my datasets and my model


Minimum configuration of the Workstation:<br />
    **GeForce GTX 1080 Ti or Higher Graphics Cards <br />
    16GB DDR4 RAM <br />
    Core i7 9th Generation Processor 8 Cores <br />
    512GB or Higher SSD** <br />

The following things you have to know about <br />
**[1. Fully Connected Layer](https://iq.opengenus.org/fully-connected-layer/)**<br />
**[2. Convolutional Neural Network](https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac)**<br />
**[3. Region-based Convolutional Neural Network](https://medium.com/coinmonks/review-r-cnn-object-detection-b476aba290d1)**<br />
**[4. Faster Region-based Convolutional Neural Network](https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8)**<br />
**[5. Overfitting and Underfitting](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)**<br />
**[6. Data Augmentation](https://towardsdatascience.com/data-augmentation-experimentation-3e274504f04b)**<br />



[modak.ipynb](modak.ipynb) is the easiest way to start.It is a step by step guide that shows how to make your own model using your own datasets.


After creating your model, you can use mdk.py to detect your object using your model. the syntex of modak.py is <br />
**python mdk.py --model_name=** **<** **path to your model**  **>** **--frozen_graph=** **<** **path to your frozen graph** **>** **--lables=** **<** **path to your labelmap** **>** **--video=** **<** **path to your video file** **>**
[![Watch the video](https://asciinema.org/a/283927.png)](https://asciinema.org/a/283927)


Output video of my model(7000+ steps) in which i used my own datasets
[![Watch the video]( https://youtu.be/ljXKc8A_sf4 )


