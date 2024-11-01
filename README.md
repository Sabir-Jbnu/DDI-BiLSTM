# DDI_Hybrid proposed model
Multiple drugs have gained attention for the treatment of complex diseases. However, while numerous drugs offer benefits, they also cause undesirable side effects. Accurate prediction of drug-drug interactions is crucial in drug discovery and safety research. Therefore, an efficient and reliable computational method is necessary for predicting drug-drug interactions and their associated side effects. In this study, we introduce a computational method based on integrating convolutional and BiLSTM networks to predict the types of drug-drug interactions. The Morgan fingerprints approach was utilized to encode the drug's SMILES, and the Tanimoto coefficient structural similarity profile-based approach was used to determine similarities. These encoded drugs were passed through convolutional and BiLSTM layers to extract important feature maps. The ReLU activation function and the dense layer were employed for feature dimensionality reduction. The last dense layer used the softmax function to classify the 86 types of drug-drug interactions. The proposed model achieved a performance of 95.38\% accuracy and 98.78\% AUC, respectively. The proposed model outperformed and surpassed all the existing state-of-the-art models.

![model_architecture](https://github.com/Sabir-Jbnu/DDI-BiLSTM/assets/124244899/4d17b005-f190-4037-a695-c01042dc682d)

# Experimental Setup details
1. Python == 3.7.0
2. Tensorflow == 2.4.1
3. keras == 2.4.3
4. spyder == 4.2.5
7. NumPy 1.19.5
8. Pandas 1.2.4
9. Scikit-learn 0.24.2
10. Scikit-image 0.18.1
11. RDKit 2023.03.2
12. All experiments were conducted with CUDA 11.2 on an NVIDIA TITAN Xp GPU with 12 GB memory.
