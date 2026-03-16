# Assignment 2: Convolutional Neural Networks for CIFAR-10 Classification

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), with 6,000 images per class.

## Project Structure

```
Assignment_2/
├── README.md                    # This file
├── 611_HW2.pdf                 # Assignment instructions
├── CNN.pdf                     # CNN reference material
├── build_cnn/                  # Source code directory
│   ├── build_cnn.ipynb         # Main Jupyter notebook with all code and outputs
│   ├── model_trained1.pt       # Trained model weights (best validation loss)
│   ├── model_trained.pt        # Additional trained model checkpoint
│   ├── cifar_data.png          # Sample CIFAR-10 images
│   ├── cat_feature.png         # Feature map visualization
│   └── data/                   # CIFAR-10 dataset (downloaded automatically)
```

## Requirements

### Python Packages
- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib

### Installation

Install the required packages using pip:

```bash
pip install torch torchvision numpy pandas matplotlib jupyter
```

Or if using a virtual environment:

```bash
# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pandas matplotlib jupyter
```

### Hardware Requirements
- GPU (CUDA) is recommended for faster training, but the code will work on CPU as well
- The code automatically detects and uses CUDA if available

## How to Run

### Option 1: Using Jupyter Notebook (Recommended)

1. Navigate to the project directory:
   ```bash
   cd Assignment_2/build_cnn
   ```

2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open `build_cnn.ipynb` in your browser

4. Run all cells:
   - Click `Cell` → `Run All` in the menu, or
   - Press `Shift + Enter` to run each cell sequentially

5. The notebook will:
   - Download the CIFAR-10 dataset automatically (first time only)
   - Train the CNN model
   - Evaluate on test data
   - Generate visualizations and feature maps

### Option 2: Using JupyterLab

```bash
cd Assignment_2/build_cnn
jupyter lab
```

Then open and run `build_cnn.ipynb` as described above.

## Code Execution Flow

The notebook is organized into the following sections:

1. **Setup and Data Loading**
   - Imports necessary libraries
   - Checks for CUDA availability
   - Loads and preprocesses CIFAR-10 dataset
   - Creates train/validation/test data loaders

2. **Model Architecture**
   - Defines the CNN architecture with:
     - 3 convolutional layers (16, 32, 64 filters)
     - Max pooling layers
     - Dropout for regularization
     - Fully connected layers

3. **Training**
   - Trains the model for 15 epochs
   - Tracks training and validation loss
   - Saves the best model based on validation loss

4. **Evaluation**
   - Tests the model on test dataset
   - Calculates per-class and overall accuracy
   - Displays test results

5. **Visualization**
   - Visualizes feature maps from the first convolutional layer
   - Shows top activating images for selected filters
   - Displays training/validation loss curves

## Model Details

- **Architecture**: Custom CNN with 3 convolutional layers
- **Input**: 32x32x3 RGB images
- **Output**: 10 classes (CIFAR-10 categories)
- **Training**: 15 epochs with SGD optimizer (learning rate: 0.01)
- **Best Model**: Saved as `model_trained1.pt` (lowest validation loss)
- **Test Accuracy**: ~70% (see notebook for exact results)

## Outputs

All outputs are stored directly in the Jupyter notebook, including:

- Training and validation loss curves
- Test accuracy results (per-class and overall)
- Feature map visualizations
- Top activating images for selected filters
- Sample test predictions with visualizations

## Trained Model

The trained model weights are saved in:
- `model_trained1.pt`: Best model (lowest validation loss)

To load and use the trained model:

```python
import torch
from build_cnn import Net  # Import your model class

# Load the model
model = Net()
model.load_state_dict(torch.load('build_cnn/model_trained1.pt'))
model.eval()
```

## Notes

- The dataset is automatically downloaded on first run (saved in `build_cnn/data/`)
- Training time depends on hardware (GPU recommended)
- All visualizations and results are embedded in the notebook
- The model returns both the first conv layer output and final classification output for visualization purposes

## Troubleshooting

1. **CUDA not available**: The code will automatically fall back to CPU training (slower but functional)

2. **Dataset download issues**: Ensure you have internet connection for first-time download

3. **Memory errors**: Reduce batch size in the notebook if running out of memory

4. **Import errors**: Make sure all required packages are installed (see Requirements section)

## Author

[Basira Daqiq]
[Course: CSCI 611]
[Assignment: 2]
