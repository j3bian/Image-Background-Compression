# Image Background Compression

### Author: Mike (JingHongYu) Bian

This project was written as part of a final project

## Motivation and Background

Uncompressed digital images are stored as collection of pixels, their resolution being the number of pixels within that collection. This makes the size of the image directly related to the number of pixels and therefore the resolution of the image. Image content is therefore completely unrelated to the size of an image; a blank white image is the same size as a densely detailed image. Is there a better solution?

<details>
    <summary>Image Compression</summary>
The JPEG (Joint Photographic Experts Group) standard is a lossy compression algorithm that provides one solution to this problem. Simply, it relies on the 2D Discrete Cosine Transform to deconstruct the image into the frequency domain and then compresses the image in the frequency domain. With more detail in the image, the more information is stored within the frequency domain, and therefore less compression can be done and the larger the file. For a blank white picture on the other hand, the information can be stored in the very low frequency range and therefore can be greatly compressed and the file is much smaller.<br>
![Figure 1] (images/figure1.jpg "Figure 1")<br>
Figure 1: Two JPEG images: Blank (left) only has size 1.27KB while Random (left) has size 22.5KB
</details>

<details>
    <summary>Ocular foveated imaging</summary>
The human eye (and many other animals) uses an imaging technique to also increase resolution of an incoming image without necessarily having to support such a high resolution. The light receptors in an eye is not evenly distributed, instead the receptors are more densely packed in the middle, called the fovea. This means the perceived image is very clear in the middle but low resolution the further out from the center, this is called foveated imaging. A simple demonstration of this phenomenon it to hold a book at arm’s length and close one eye, place a thumb on the page and with the open eye, stare directly at your thumb and try to read the words on the page with your peripheral vision.
A simple estimate of the number of light receptors in a human eye gives us an approximation of 5 million receptors, translating to a 5 megapixel resolution. However, using a foveated imaging technique and moving one’s to translate the center of the image, the human eye mimics a resolution of 576 megapixels .<br>
![Figure 2] (images/figure2.jpg "Figure 2")<br>
Figure 2: Rendered simulation of foveated imagining. The center remains very highly detailed while the peripherals are blurred.
</details>

<details>
    <summary>Background compression</summary>
The foveated imaging technique relies on a central point of focus at all times. With rotatable eye humans can rely on this ability to maximize resolution, however this does not work for still images. Still images often times have off center regions of interest (foreground objects) therefore we cannot use centered foveated imaging on still images.
If we had the ability to locate the regions of interest (the foreground objects), then we can compress the unimportant background while maintaining the detail of the important foreground areas. This compression method can be overlapped on to the JPEG compression algorithm to provide the most compact compression without sacrificing the details in the image.
</details>

## Abstract

The focus of this project is in on the foreground identification portion. Foreground is not quite the appropriate term for actual regions that are eventually found, although it does stem from the idea of foveated imaging centered on some “foreground”. Instead of foreground, we will be finding objects of certain sizes, these will be the details to keep. Once objects of a wide enough range are found, the rest of the image is all considered background, or unimportant. Using this method, we will extract the areas of high detail.
This idea stems from the logical assumption that pixels in an image has some spatial relationship to one another and therefore by identifying regions of similar pixels, we are identifying objects. Once we have identified the objects of interest, then we can represent those objects in a lower dimensional manifold.
To identify these object (or regions of detail), we make another assumption that typical objects are in regular shapes (triangle, rectangle and ellipse). A model will be trained to identify these shapes and when they are found, the pixels in those shapes will be marked as “detail”.

## Algorithm

### Models

A Variation Autoencoder  (VAE) is trained on a dataset of shapes as well as a dataset of randomly generated images (noise). Then a supplementary model, a discriminator, was trained off the output of the encoder portion to identify noise from shapes. To simplify the models and data generation, the models will only accept 1 channel of color (typically greyscale).
A VAE is used since we make the assumption that objects of interest belong to 3 categories of distinct shape: Triangle, Rectangle and Ellipse. We therefore know that latent variables must exist for these shapes and therefore a VAE is used. Furthermore we want the VAE to be resilient to noise, therefore training inputs have noise added on top while the targets were noise free. Theoretically, the VAE will be able to regenerate the shape in the input image without the noise. 
The structure of the VAE made up of an encoder and a decoder. The encoder starts with three sets of Convolution + Pooling layers followed by a fully connected layer leading to the latent variables layer, the decoder is the same but in reverse with upsampling layers replacing pooling layers. The decision for three sets of Conv + Pool layers was made since the feature for differentiating shapes is not very difficult so it should not require a very deep network, however a feature at the deepest convolution layer should span the width of at least half the input size. With a default input size of 16x16, this was determined that 3 should be enough; trials confirmed this hypothesis.

![Figure 3] (images/figure3.jpg "Figure 3") \
Figure 3: VAE model structure (input feeds from bottom up)

![Figure 4] (images/figure4.jpg "Figure 4") \
Figure 4: Encoder and discriminator model structure (input feeds from bottom up)

No dropouts were used since noise was artificially generated to reduce overfitting. Also, the plan was to train on a training set of size ~ 100000 shapes with another 100000 of empty inputs. A rough estimate of unique shapes to fit in a 16x16 grid of sufficient size gives < 100000, therefore overfitting is highly unlikely when the training set is larger than the actual number of possible shapes.
The classification of the input into indicate if it contains a shape or is just an empty input is simple since this information is already stored in the latent variables of the VAE. The supplementary model is a simple one node fully connected layer connected to the output of the encoder trained to classify if the input contains a shape. Note that when this supplementary layer is training, the encoder weights should not change.

### Dataset

A dataset of shapes is generated such that the shapes have randomly generated orientation, position and skew, however must be a big enough to identify but not encompassing the entire image (a fixed identifiable size). The shapes are then set to be a specific shade (in a range of 1..255 out of 255) and a gradient is applied over it to simulate lighting effects. Small objects are then laid over top of the shape to simulate objects in front of the shape. Then background is generated as either plain color or a random background (noise) and shading is applied over the whole image. Lastly noise is applied to the whole image and blur is applied to simulate high ISO noise and mis-focused images. An equal number of empty images (with the same noise addition process applied) is added to the dataset for an even number of negative (no shape) samples. These techniques trains the VAE to be more noise tolerant. 
Special credit must be given to the creators  of the Baby AI Shapes Dataset . The shape generation code was copied and altered from their dataset. 

### Search Algorithm

Given an image, the search algorithm simply scans through the image using the encoder and discriminator to identify shapes using a specified window size. A tile made from such window is then resized to fit the encoder input size and passed through the encoder. If a shape is found, the decoder reconstructs the region where the shape is using the encoding and adds that specific area to a mask. The mask indicates all the “foreground” (detailed) regions of that specific window size.
Many window sizes may be used to identify many masks for the foregrounds (since each mask only masks shapes of a particular size). A combination of {1/2,1/4,1/8}  and the mask added together using an OR-like operation was found to work the best, however different images may require different window sizes. Finally, the background is blurred while the foreground remains details and the image is compressed into a JPEG image. The blurring of the background makes the background more compressible using JPEG.

## Performance

The compression of the image some sample images show an approximately 50% reduction in size from the original image size. Although the alterations to the image is noticeable, the details on the important objects remain unchanged. Furthermore, the objects in the images are recognizable however, text sometimes is lost as it is treated as a “texture” on the background.

![Figure 5] (images/figure5.jpg "Figure 5") \
Figure 5: Original Sample

![Figure 6] (images/figure6.jpg "Figure 6") \
Figure 6: Background Compression result of a sample image. Some roof tiles as well as parts of the street are visibly blurred

![Figure 7] (images/figure7.jpg "Figure 7") \
Figure 7: JPEG compressed image to a similar size as the Background Compressed image

A zoomed in view shows that the Background Compression method preserves the file details while the JPEG compression losses some finesse.

![Figure 8] (images/figure8.jpg "Figure 8") \
Figure 8: Zoomed into the 3 images. Left: Original, Middle: Background Compression, Right: JPEG compression

## Conclusion

Background Compression is very viable compression strategy for image compression. The determination of which pixels are the background and which are the foreground is a hard problem. Using a VAE to identify foreground objects of varying sizes is one solution to the problem; the examples shows a proof of concept but further improvements must be made to make the models more robust. Furthermore, the thresholds of the VAE can be tuned to reduce errors and provide accuracy. Lastly the algorithm uses an iterative method of windowing and layering the foreground masks, this is slow and causes sharp edges between foreground and background. A future improvement to this portion of the algorithm can greatly reduce the shape edges and also improve the overall performance by eliminating some errors form the VAE output (false positives and negatives).

## Image credit

Images were taken from https://pixabay.com/ \
sample.jpg by : [MemoryCatcher](https://pixabay.com/photo-3317984/)\
sample2.jpg by : [realworkhard](https://pixabay.com/photo-190432/)

## Dataset credit

All credit for the data generation goes to the creators of the [Baby AI Shape Datasets](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/BabyAIShapesDatasets)