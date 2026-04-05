# Movie Sentiment Classifier — Deep Learning with TensorFlow

Neural Network built with TensorFlow and Keras to classify
IMDB movie reviews as Positive or Negative.

## Results
- Test Accuracy : 87%  ← update with your actual number after running
- Architecture  : Embedding → GlobalAveragePooling → Dense → Dropout → Sigmoid
- Dataset       : IMDB 50K reviews (built into TensorFlow — no download needed)
- Epochs        : 10

## What This Project Covers
- Deep Learning with TensorFlow and Keras
- Word Embedding layer for text representation
- Neural Network architecture design (Dense layers, ReLU activation)
- Dropout regularization to prevent overfitting
- Training vs validation accuracy and loss monitoring
- Binary classification with sigmoid activation

## Tech Stack
Python, TensorFlow, Keras, NumPy, Matplotlib

## Project Structure
```
movie-sentiment-tensorflow/
├── src/
│   └── sentiment_nn.py   ← run this
├── outputs/              ← charts saved here automatically
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install TensorFlow
```bash
pip install tensorflow
```

### 2. Run the model
```bash
cd src
python sentiment_nn.py
```

No dataset download needed — IMDB data loads automatically via TensorFlow!
