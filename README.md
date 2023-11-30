# weighted bad semi-supervised generative adversarial networks with individual-relational consistency regularization (WB-SGAN-IRC). 
A Novel Individual-Relational Consistency for Bad Semi-Supervised Generative Adversarial Networks in Skin Lesion Diagnosis 

## Table of Contents
- [Authors](#authors)
- [Abstract](#abstract)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Authors

- Mohammad Saber Iraji
- Jafar Tanha (Corresponding author: tanha@tabrizu.ac.ir, jafar.tanha.pnu@gmail.com)
- Mohammad-Ali Balafar
- Mohammad-Reza Feizi-Derakhshi

This work was conducted by researchers from the Department of Computer Engineering, Faculty of Electrical and Computer Engineering, University of Tabriz, Tabriz, Iran.

## Abstract

The classification of skin lesions using machine learning (ML) has attracted considerable research interest in the field of skin cancer diagnosis. Due to the small number of labeled samples and the high cost of manual labeling, semi-supervised learning with pseudo-labels for unlabeled samples has been proposed to improve the classification of skin lesions. Despite its potential benefits, the classification of skin lesions via semi-supervised learning is still plagued by several issues: 1- bias of classification predictions toward the majority class, 2- incorrect pseudo-labels, particularly for low-confidence unlabeled samples, and 3- ineffective consistency regularization in low-density areas due to neglecting the connections between data samples. To this end, a novel framework is proposed, termed weighted bad semi-supervised generative adversarial network with a new individual-relational consistency regularization loss function, to aid physicians in early skin cancer detection. Our approach employs a bad generator, classifier, and discriminator to handle individual-relational consistency regularization in the latent space of bad fake data. Additionally, a weighted cross-entropy and weighted inversed cross-entropy loss function are used to reduce the bias of the classifier's predictions on bad images. The notable improvement in model generalization is the significant impact of incorporating individual-relational consistency regularization for bad fake samples and considering connections between data samples. The enhanced bad generator accurately generates fake samples with decision boundary information and improves the pseudo-labeling of unlabeled samples. We evaluated our proposed approach on skin lesion images with limited and imbalanced labeled data. The evaluation results indicate that the method improves the performance of skin lesion classification, achieving F1 index values of 55.55, 61.60, and 64.52 with 5%, 10%, and 20% training data, respectively, compared to state-of-the-art (SOTA) methods. 

## Key Features
Skin lesion classification; Informative fake samples; Low-confidence samples; Relational consistency regularization; Weighted inversed cross-entropy
## Installation and Usage

To use the WB-SGAN-IRC algorithm, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Configure the training parameters and dataset paths in the provided configuration file.
4. Evaluate the trained model using `RCRWBSGAN.py`.


## Results

The evaluation results indicate that the method improves the performance of skin lesion classification, achieving F1 index values of 55.55, 61.60, and 64.52 with 5%, 10%, and 20% training data, respectively, compared to state-of-the-art (SOTA) methods. 


## Contributing

Contributions to this project are welcome. If you have any suggestions, improvements, or bug fixes, please submit a pull request or open an issue on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).
