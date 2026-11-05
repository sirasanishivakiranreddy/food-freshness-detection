# Food Freshness Detection Using CNN

A Python project for classifying food images as fresh or rotten using a CNN built with TensorFlow/Keras, with a Streamlit interface.

## Dataset layout

Place images in the matching folders. Fruit type is not a class: apples,
bananas, and oranges all go into either `fresh` or `rotten`.

```text
dataset/
├── train/
│   ├── fresh/
│   └── rotten/
├── validation/
│   ├── fresh/
│   └── rotten/
└── test/
	├── fresh/
	└── rotten/
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python --version
pip install -r requirements.txt
```

## Train and run

Prepare the selected apple, banana, and orange images from the source archive:

```bash
python prepare_dataset.py /path/to/Original\ Image.zip
```

The script uses equal counts for each fruit/freshness category, then creates
deterministic 70/15/15 train, validation, and test splits. It keeps source
images separate across splits and does not augment validation or test images.
If the source does not contain enough examples for the 3,000-image target, it
uses the largest balanced subset available without duplicating images.

The original images come from Sultana, Jahan, and Uddin, “Fresh and Rotten
Fruits Dataset for Machine-Based Evaluation of Fruit Quality” (Mendeley Data,
2022, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[DOI: 10.17632/bdd69gyhv8.1](https://doi.org/10.17632/bdd69gyhv8.1)).

After preparing the folders, train and evaluate the CNN:

```bash
python train.py
```

The best validation model is saved as `model/food_freshness.keras`. Training
also saves `model/training_accuracy.png` and `model/confusion_matrix.png`, and
prints test accuracy plus a class-wise precision/recall report. In the current
run, test accuracy was 94.44% on 180 held-out images (90 fresh, 90 rotten).
Results depend on the dataset and are not a guarantee of performance on new
images. Start the upload interface with:

```bash
streamlit run app.py
```

The app predicts **Fresh** or **Rotten** for an uploaded image. Keep the fruit
types represented in both labels and splits; model quality depends on the
quality and balance of the collected images.
