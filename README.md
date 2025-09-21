# Utilizing Temporal Dependencies and Semi-Supervision to Improve Combined Intrusion Detection

This repo contains the code for my master thesis. 

## Abstract

Intrusion detection is an essential part of protecting today’s computer systems.
Recent work has shown that host-based \acp{ids} can benefit from directly including network data in the decision-making process.
In this thesis, we build upon this work and further enhance these systems by exploiting multiple properties of the underlying data.
First, we aim to utilise the temporal structure of our data by applying Transformer models.
We begin with a theoretical analysis of the application of the Transformer model, identifying key challenges that prevent straightforward application to anomaly-based intrusion detection.
During our study of the Transformer models, we found a misleading classification of \acp{ids} developed using supervised \acl{ml} models, including those utilising the Transformer, in anomaly-based systems.
We shift our focus to analyse this misclassification to understand current research better.
We find that precise classification boundaries cannot be drawn in anomaly detection settings, leading us to conclude that those behave more like signature-based detection methods.
Overall, we provide key insight into supervised model behaviour to enable the future transfer of Transformer models into anomaly-based settings.
The second data property we leverage is the availability of previously obtained malicious samples during design time.
We develop a semi-supervised training procedure to include these malicious samples in training and improve the anomaly detection capabilities of \iac{ids}.
Our method consists of an unsupervised pretraining stage using only benign data, followed by a supervised finetuning that maximises the anomaly score for malicious samples.
Evaluation results showed that our method could improve combined intrusion detection under the right conditions, primarily by reducing the False Positive Rate, which results in favourable behaviour towards practical application compared to other approaches.

## Structure

The repo is structured as followed:

- ```cids```: contains the core code for models, handling of preprocessed data, anomaly calculation and training procedures
- ```config```: Here the config files for different experiments and model types are saved
- ```executables```:  Helpful .sh executables
- ```experiments```: Code to run different experiments
- ```notebooks```: All jupyter notebooks
- ```preprocessing```: Handling of data preprocessing for the OpTC Dataset
- ```scripts```: Other useful scripts

