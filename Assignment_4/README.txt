Neural Style Transfer Using VGG19

This project implements the neural style transfer algorithm from "Image Style Transfer Using Convolutional Neural Networks" by Gatys et al. (CVPR 2016) using PyTorch.

The method takes a content image and a style image, then generates a new image that preserves the content structure while adopting the artistic style.

Requirements

Install the required Python packages before running the notebook:

pip install torch torchvision pillow matplotlib numpy requests

How It Works

1. A pre-trained VGG19 network extracts features from the content image, style image, and a target image.
2. Content features are captured at layer conv4_2 (deep layer that encodes object structure).
3. Style features are captured at layers conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1 using Gram matrices to represent texture correlations.
4. The target image starts as a copy of the content image and is iteratively updated to minimize a total loss that combines content loss and style loss.
5. After 2000 optimization steps, the target image has the content of one image rendered in the style of the other.

How to Run

1. Open Style_Transfer_Exercise.ipynb in Jupyter Notebook or VS Code.
2. Place your content and style images in the Assignment_4 folder.
3. Update the image file names in the loading cell if using different images.
4. Run all cells from top to bottom.
5. The training loop displays intermediate results every 400 steps and prints the total loss.
6. The final cell displays the content image alongside the stylized result.

Files

Style_Transfer_Exercise.ipynb - Main notebook implementing neural style transfer.
audrey_content.jpg - Content image (provides structure and layout).
farida_style.png - Style image (provides artistic texture and color patterns).
Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf - Original paper by Gatys et al.

Hyperparameters

content_weight (alpha) = 1
style_weight (beta) = 1e6
Learning rate = 0.003
Optimizer = Adam
Steps = 2000
Style layer weights: conv1_1=1.0, conv2_1=0.8, conv3_1=0.5, conv4_1=0.3, conv5_1=0.1
