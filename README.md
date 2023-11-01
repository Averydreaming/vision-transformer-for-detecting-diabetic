# vision-transformer-for-detecting-diabetic
Retinal screening plays a crucial role in the early detection and timely treatment
of diabetic retinopathy. Convolutional Neural Networks (CNNs) have emerged
as the prevailing model for computer vision tasks. Transformers [ 3], initially
introduced by Google in 2017 as a classic model for natural language processing
(NLP), have gained attention in the computer vision domain,which utilize the
self-attention mechanism and do not rely on sequential structures like Recurrent
Neural Networks (RNNs). This design allows for parallel training and enables
the model to capture global information. Recent studies have shown that Vision
Transformers[1 ] can achieve comparable or even superior performance in image
classification tasks.However, whether this approach can achieve good performance
in the domain of medical image recognition, where datasets are generally smaller,
remains an open question.
In this study, we applied deep transfer learning using Vision Transformers to
automatically classify any diabetic retinopathy lesions present in retinal images,
determine the progression of diabetic retinopathy, and proposed optimization
strategies. We utilized pretrained Vision Transformers (ViT) for transfer learning.
Considering the significant impact of false positives and false negatives in medical
diagnosis, we assigned higher weights to individuals who were misdiagnosed with
the disease during model training, aiming to minimize such occurrences. This
approach is known as cost-sensitive learning. The proposed method demonstrated
higher accuracy, sensitivity, and specificity, surpassing the classification accuracy
of CNNs.
Throughout this research, I acquired valuable knowledge in important machine
learning theories such as self-attention mechanisms, transfer learning, and pre-
training. Additionally, we conducted a comparative analysis to understand factors
influencing Vision Transformer performance and discussed the advantages and
disadvantages of Vision Transformers compared to CNNs.
