# FIRE-SMOKE-CLASSIFFICATION
 This project develops an image classification system using deep learning to detect fire, smoke, and non-fire categories in images. It combines a custom CNN and pre-trained ResNet50 model through transfer learning. The system offers high accuracy, real-time detection, and can be applied in safety and fire prevention applications.
Overview
This project implements a deep learning solution for detecting fire and smoke in images. It uses Convolutional Neural Networks (CNN) and the ResNet50 model for transfer learning to classify images into three categories: fire, smoke, and non-fire. This solution can be utilized for real-time fire detection applications, improving safety and emergency response times.

Features
Image Classification: Classifies images into three categories: fire, smoke, and non-fire.
Deep Learning Model: Custom CNN architecture combined with pre-trained ResNet50 for efficient learning.
Transfer Learning: Leverages the power of pre-trained models to boost performance on the fire and smoke detection task.
Accurate Detection: High classification accuracy with a focus on real-world applications.
Requirements
To run the project, you need the following Python packages:

Python 3.12
TensorFlow 2.18.0
Keras
NumPy
OpenCV
Matplotlib
Scikit-learn
You can install the required libraries by running the following command:

Copy code
pip install -r requirements.txt
Dataset
The dataset for this project consists of images categorized into three classes:

Fire: Images with fire.
Smoke: Images with smoke.
Non-Fire: Images without fire or smoke.
You can download the dataset from Dataset Source. Ensure the dataset is organized with each class stored in its respective folder.

Model Architecture
The model consists of:

A custom CNN for feature extraction.
A ResNet50 model for transfer learning, which enhances performance by fine-tuning the pre-trained model on fire and smoke data.
The project uses the following layers and techniques:

ResNet50: Pre-trained on ImageNet and used as a base model.
Fine-Tuning: Fine-tuning of the model with our specific fire and smoke dataset.
Fully Connected Layers: To map the extracted features to the final prediction classes.

Prediction Results - 
![image](https://github.com/user-attachments/assets/1e08970a-1d11-4d98-a6bd-c344d41837dd)
![image](https://github.com/user-attachments/assets/d4ef6cf9-3969-4689-8e98-b4ddd549c555)
![image](https://github.com/user-attachments/assets/38305486-d2e2-45ce-b4a9-544cbec14ecd)



Contact
For any questions or contributions, please feel free to open an issue or reach out to [sumit.thakur12492@gmail.com].
