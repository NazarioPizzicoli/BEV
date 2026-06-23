# Dataset

## Input:
- **BEV MAP**: Features fused taken from GaussianCar. More specifically after the Feature Pyramid Network. [128,200,200]
- **QA Dictionary**: Question and answer (text) from NuScenes-QA.
- **Glove** :Global Vectors for Word Representation. It's based on co-occurrence (min the error between the word vector's dot product and the co-occurrence counts logarithm) and i use the one made by 6B of tokens,400K words, 300d vectors (Wikipedia 2014 + Gigaword 5). Embeddings for word.

## Output(Dataset):
- **BEV Features**
- **Question Embeddings**
- **Answer (token)**

# MODEL
## 1. ENCODER BEV:
1. **Model 1**:

    [B,128,200,200]->AdaptiveAvgPool2D->[B,128,20,20]->Flatten+Transpose->[B,400,128]->Linear->[B,400,512]

2. **Model 2**:


## 2. ENCODER TEXT: 

    Glove['domanda']->[B,30,300]->AdaptiveAvgPool2D->[B,128,20,20]->Flatten+Transpose->[B,400,128]->Linear->[B,400,512]

## 3.  MCA
- MultiHeadAttention for bev
- MultiHeadAttention for text
- MultiHeadAttention on text
- Norm1
- Norm2
- Norm3

## 4. OutputHead
- Linear

