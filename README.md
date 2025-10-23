# RetinAI-ViT

Retina AI and Vision Transformer

Diabetic Retinopathy (DR) is a leading cause of blindness among working-age adults. Early detection and treatment can significantly reduce the risk of vision loss. This project aims to automate the DR grading process using a state-of-the-art Vision Transformer model. The model classifies retinal images into five severity stages, from "No DR" to "Proliferative DR," providing a powerful tool for screening and diagnosis assistance.

This model leverages the power of pre-trained **BEiT2**, enhanced  with **Attention Mehanism** to focus on critical regions of the retina for more accurate predictions.

Key Features
 - **State of the Art Core**: Utilizes the powerful **BEiT2** model from huggingface, pre-trained on massive dataset, for superior feature extraction.
 - **Attention Mechanism**: Implements a `Multi-Head Attention` layer on top of the BEiT2 embeddings to dynamically weigh the importance of different image patches, mimicking how a clinician might focus on specific retinal areas.
 - **Class Imbalance Handling**: Employs a `WeightedRandomSampler` during training to oversample minority classes, preventing the model from being biased towards the majority class.
 - **Data Augmentation**: Applies random flips, rotations and color jitters to the training data to improve model generalization and robustness.
 - **Comprehensive Evaluation**: Measure performance using **Accuracy**, **F1-Score**, **Precision** and **Recall**.
 - **Visualization**: Generates a confusion matrix and training history plots for clear and insightful model analysis.

 ### Model Architecture

 The data flow is as follows

 `Input Image  BEiT2 Preprocessor  BEiT2 Base Model  Patch Embeddings  Multi-Headed Attention  Classifier Head  Logits`

 1. BEiT2 Base (Feature Extractor): A pre-trained BeitModel is used as the backbone. The majority of its layers are frozen, and only the final two encoder layers are fine-tuned. It takes a preprocessed image and outputs a sequence of patch embeddings.
 2. Multi-Head Attention Layer: This layer takes the patch embeddings from BEiT2 as input. It learns to re-weigh these patches, allowing the model to focus on the most informative regions of the image (e.g., areas with microaneurysms, hemorrhages, or exudates).
3. Classifier Head: A feed-forward network with GELU activations and Dropout layers takes the attended [CLS] token embedding and maps it to the final 5-class output logits.