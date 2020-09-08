# Semantic-segmentation-with-toy-data
In this work, I learnt how to use the UNet model to do the semantic segmentation in my own generated datasets. 

(I) Briefly introductions of my own generated datasets.

1/ I assumed a simple Universe which only contains two kinds of stars (blue and red). 

2/ The flux relations between these two kinds of stars are very simple. Blue stars are twice brighter than red stars in color-b, while red stars are twice brighter than blue stars in color-r. 

3/ I generate 50 images (256x256) with 10 blue and 10 red stars in each of them. The fluxes of those stars are inserted randomly and conserved with the assuming flux relations.

3/ I further generate the well-labeled pixel-wise mask for each star in the 50 generated. I label the blue stars as [1,0], red stars as [0,1], and background as [0,0] in the mask images. 

One example of my generated training dataset (see below)
![image](https://github.com/bobby891018/Semantic-segmentation-with-toy-data/blob/master/Figures/train.png)


(II) Briefly introductions of the training model (UNet)

In this work, I adopted UNet architecture from the python tensorflow.keras functional API. 

The model is trained for 500 epochs, while the iterations will stop when there is no improvement in 5 steps. The output from the architecture is a 256x256x2 image which represents the mask that should be learned. Softmax activation function had been used in the last layer for performing a classification task. I adopted the categorical cross-entropy as a loss function for the training.

(III) Results

Use the trained model to do segmentation on test datasets (see below)
![image](https://github.com/bobby891018/Semantic-segmentation-with-toy-data/blob/master/Figures/results.png)
