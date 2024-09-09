# System of generating new faces


## Description
This project implements a generative adversarial network (GAN) to create realistic images of faces. GAN consists of two neural networks — a generator and a discriminator, which are trained together to generate images that are difficult to distinguish from real ones. 
The Celeba dataset was used in this project (https://www.kaggle.com/datasets/jessicali9530/celeba-dataset ). 
The project implements the ability to track the progress of model learning in dynamics using the example of an image canvas using animated output to the browser.
The project was developed and supplemented as part of a large project within the framework of image generation research.


## Technologies used
- **Python 3.8+**
- **PyTorch**
- **torchvision.transforms** (to apply transformations to images)
- **NumPy**
- **Matplotlib** (for image visualization)
- **PIL** (to work with images)
- **webbrowser** (to display the animation)

  
## Scheme of the program operation

The program is divided into 3 logical parts:

1. Initialization of the generator and discriminator model
2. Training of an adversarial neural network
3. Dynamic image generation using a trained neural network

**The first** part is solved using convolutional neural networks. One network learns to create fake images, and the other learns to distinguish fake images from real ones, due to which the overall quality of the model increases. **The second** part is the model learning pipeline, which learns using the loss function nn.BCELoss(). To ensure the reproducibility of the solution, initialization with the initial values of both the generator model and the discriminator model is used. **The third** part is logging the canvas of the reproduced images of the generator in the process of training the model and creating animation.

In order to launch a project, you need to:
1. Clone repository
```bash
git clone https://github.com/Samirml/FaceGeneration
cd gan-face-generation
```
2. Create a virtual environment and activate it
Linux:
```bash
python3 -m venv env
source env/bin/activate
```
Windows
```bash
python -m venv env
env\Scripts\activate
```
3. Install dependencies
```py
pip install -r requirements.txt
```
4.Project Launch (to train and generate images)
```bash
python generate_faces.py
```

## License
The idea of the project was taken from Karpov.Courses (https://karpov.courses/deep-learning?_gl=1*gvc6ll*_ga*NDI1MzY4NTU3LjE3MjM5NzU4OTE.*_ga_DZP7KEXCQQ*MTcyNTg3MzAyNi4xMTYuMC4xNzI1ODczMDI2LjYwLjAuMA..).

## Авторы и контакты
To contact the author, write to the following email: samiralzgulfx@gmail.com


