# Fixing-OOM-in-TensorFlow
## Summary
Tired of seeing 'OOM'? Here's a simple way to input large datasets into a TensorFlow model with data chunking and transfer learning.

Take a large dataset, for example, the Fashion MNIST dataset contains 28,000 images. The images are only 28 x 28, but what if these images were 512 x 512? Most real life dataset have a resolution much greater than 28 x 28. So how do you train this on your GPU, even on a Google Colab GPU, if you only have a few GB of VRAM? And, a more overlooked important aspect, how do you even load a dataset this large into your RAM?

## How it Works
This notebook explains how to load a large dataset in chunks, and then how to train using chunks of those chunks.
