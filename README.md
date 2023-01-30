# Fixing-OOM-in-TensorFlow
## Summary
Tired of seeing 'OOM'? Here's a simple way to input large datasets into a TensorFlow model with data chunking and transfer learning.

Imagine you had a dataset of 10,000 images of size 512 x 512. By today's standards, this isn't a particularly large dataset, but you probably wouldn't be able to train a dataset this big on something like Google Colab with its 15 GB VRAM, in fact, you may not even be able to load the dataset within its 12 GB system RAM...

## How it Works
This notebook explains how to load a large dataset in chunks, and then how to train these chunks using transfer learning.

### (1/2) Loading a Dataset in Chunks
This GitHub project contains a dummy dataset. It contains 6,000 images, all of size 2 kB. This might not seem like a lot of images for a large dataset, but even with a dataset this small, we can't increase the image size of these single channel, greyscale images too far past 1,500 x 1,500, we can see how common resources like Google Colab, with its 12 GB of system RAM, will not be able to handle this dataset.

The effect on system RAM when loading 6,000 2 kB images at different image sizes. System RAM includes approximately 2.3 GB of memory used in loading libraries needed to load the dataset.
| Image Size | System RAM |
|-----------|-----------|
|  256 x 256  | 2.6 GB  |
|  512 x 512  | 3.3 GB  |
| 1024 x 1024 | 6.3 GB  |
| 1536 x 1536 | 11.1 GB |
| 2048 x 2048 | > 12 GB |

Some datasets are high resolution and contain dozens of thousands of images, so it's pretty easy to fill up the 12 GB of system RAM available on Google Colab. So what's the solution? We can simply load the batches during training, in our code, it's as easy as going through the dataset directory like this: ```for fname in sorted(os.listdir(data_dir))[lower:upper]:```, where the lower and upper variables increment each epoch and loop back around once the end of the dataset is reached.

### (2/2) Transfer Learning
Then, after training for one chunk of data, we can save the model, load the next chunk of data, reload the model weights, and continue training on the next chunk of data.
