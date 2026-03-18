###### Official implementation of the Research Article "Leveraging Dual Transfer Learning and Attention-based Pooling for Lung and Colon Cancer Histopathological Image Classification".

###### 

Python   3.8+
PyTorch  2.0+
License  MIT
Status   Completed
---





###### 

###### The final structure should look like this:

----------------------------------------------------------------------

LungColonCancer\_Classification/
├── data/
│   └── lung\_colon\_image\_set/
│       ├── lung\_aca
│       ├── lung\_n
│       ├── lung\_scc
│       ├── colon\_aca
│       └── colon\_n
├── models/
├── utils/
└── train.py
---

###### 



###### 

###### 📝 Abstract

###### --------------------------------------------------------------------------------------------------

###### Cancer deaths caused by lung cancer remain among the dominant causes of cancer mortality worldwide. This paper presents a new hybrid architecture that improves lung and colon cancer diagnosis through better classification of histopathological images. It combines Dual Transfer Learning with Neighbor Feature Attention-based Pooling (NFP) to ease the challenges involved in medical image processing. The proposed method achieves an accuracy of 96.3% on the LC25000 dataset, surpassing current state-of-the-art methods.





###### 🚀 Key Features

###### --------------------------------------------------------------------------------------------------

###### &nbsp;	i) Dual Transfer Learning: Utilizes DenseNet-169 and InceptionResNet101-V2 in parallel to capture complementary features (feature propagation and multi-scale features).

###### &nbsp;	ii) Neighbor Feature Attention-based Pooling (NFP): Replaces standard Global Average Pooling to preserve spatial dependencies and tissue context, which are critical for histopathological analysis.

###### &nbsp;	iii) Feature-level Fusion: Concatenates high-level features from both backbones for robust classification.

###### &nbsp;	iv) High Performance: Achieves state-of-the-art results on the LC25000 dataset.





###### 🛠️ Installation

###### -----------------------------------------------------------------------------------------------------

1. Clone the repository:
   git clone https://github.com/YOUR\_USERNAME/LungColonCancer\_Classification.gitcd LungColonCancer\_Classification
   ---
2. ###### Create a virtual environment (Recommended):

&nbsp;	python -m venv venv
	source venv/bin/activate  # On Windows use `venv\\Scripts\\activate
---

3. ###### Install dependencies

###### pip install -r requirements.txt



###### 

###### 📂 Dataset Preparation

---------------------------------------------------------------------------------------------------------------------------------

This project uses the LC25000 Dataset (Lung and Colon Histopathological Images).

1. ###### Download the dataset from the original source link available in the Acknowledgements section.
2. ###### Extract the contents.
3. ###### Place the lung\_colon\_image\_set folder inside the data/ directory of this project.

###### 

###### 

###### 🏋️ Training

-------------------------------------------------------------------------------------------------------------------------------------

###### To train the model from scratch using the parameters described in the manuscript (Batch size=32, LR=0.001, Epochs=50):

###### 

###### python train.py

###### The training script will:

Automatically split the data into 80% Train, 10% Validation, 10% Test.
Apply Data Augmentation (Rotation, Flipping, Zoom).
Save the best model weights as best\_model.pth.
---

###### 

###### 🧪 Evaluation

------------------------------------------------------------------------------------

To evaluate the trained model and generate the Confusion Matrix and ROC Curve:

###### python evaluate.py

###### Outputs will be saved in the results/ directory:

results/confusion\_matrix.png
results/roc\_curve.png
---

###### 

###### 🙏 Acknowledgements

------------------------------------------------------------------
The LC25000 Dataset creators:
---

###### Borkowski, A. A., Bui, M. M., Thomas, L. B., Wilson, C. P., DeLand, L. A., \& Mastorides, S. M. (2019). Lung and colon cancer histopathological image dataset (lc25000). arXiv preprint arXiv:1912.12142.

###### Download the Dataset from here:

###### https://github.com/tampapath/lung\_colon\_image\_set/blob/master/README.md

