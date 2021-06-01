**Semi Supervised Intent classification using Self-Training Technique with different Data Augmentation methods:**

Apprroach:
The second method we employ under semi-supervised intent classification is using the self-training semi-supervised technique with data augmentation, to classify labels. To encode each sentence, the methodology uses modern transformer-based (XL-Net) models, and several data-augmentation approaches to parse the given sentences as enhanced data. We used data augmentations to standardize the label distributions and computed supervised loss throughout the training procedure for labeled phrases. We investigated self-training for unlabeled sentences by treating low-entropy predictions as pseudo labels and high-confidence predictions as labeled data for training. We also added consistency regularisation as an unsupervised loss following unlabeled data augmentations, based on the notion that the model should predict similar class distributions with original unlabeled sentences and supplemented phrases as input. With the help of a set of experiments, we demonstrated that our system performs extremely well in terms of F1-score and accuracy in predicting labels for the entire dataset.


![Screenshot (247)](https://user-images.githubusercontent.com/26361255/120319481-b0e1c580-c2fe-11eb-8f60-ef2b0e8eb730.png)

Folder Structure:
1. processed_data: This folder contains two sub-folders - input_folder and output_folder. input_folder contains the initially labeled and unlabaled files for both the intents along with the final augmented labeled data files (using BERT model data augmentation technique). output_folder contains the complete output labeled datatset for both the intents.
2. Data_Augmentation_Methods.ipynb: This file is a Jupyter notebook where three different state-of-the-art data augmentation techniques have been used for data augmentation. after manually checking the quality of data augmentation, the data augmented by the best technique has been chosen for further use.
3. Data_Preparation.ipynb: In this Jupyter notebook, data cleaning and preprocessing has been performed and finally, it outputs four files - labeled and unlabaled for both the intents. The labeled output file will then be used to perform data augmentation.
4. SSL_Data_Augmentation_Technique_Q1.ipynb:  This Jupyter notebook, is the main file where the system to predict intent 1 has been implememnted.
5. SSL_Data_Augmentation_Technique_Q2.ipynb:  This Jupyter notebook, is the main file where the system to predict intent 2 has been implememnted.

