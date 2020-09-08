# Semantic-segmentation-with-toy-data
In this work, I learnt how to use the Unet model to do the semantic segmentation in my own generated datasets. 

(I) About my own generated datasets.

1/ I assumed a simple Universe which only contains two kinds of stars (blue and red). 

2/ The flux relations between these two kinds of stars are very simple.

Blue stars are twice brighter than red stars in color-b, while red stars are twice brighter than blue stars in color-r. 

3/ I generate 50 images (256x256) with 10 blue and 10 red stars in each of them. 

The fluxes of those stars are inserted randomly and conserved with the assuming flux relations.

3/ I further generate the well-labeled pixel-wise mask for each star in the 50 generated. 

I label the blue stars as [1,0], red stars as [0,1], and background as [0,0] in the mask images. 

