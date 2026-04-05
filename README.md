# Sentiment Analysis for Movie reviews — TensorFlow Neural Network
Built a Neural Network using TensorFlow and Keras to classify
IMDB movie reviews as Positive or Negative.
First time working with deep learning and Embedding layers.

# Results
Test Accuracy : 85.45%
Test Loss     : 0.35
Epochs        : 10
Dataset       : IMDB 50K reviews (built into TensorFlow)

# Tech Stack
Python, TensorFlow, Keras, NumPy, Matplotlib

# How to Run
bashpip install tensorflow matplotlib numpy

cd src
python sentiment_nn.py

# Model Architecture
Embedding (10000 words → 32 dimensions)
GlobalAveragePooling1D
Dense (64, ReLU)
Dropout (0.3)
Dense (32, ReLU)
Dropout (0.3)
Dense (1, Sigmoid)

# What I Learned
Embedding layer was completely new to me. It converts each word
into a 32-dimensional vector that captures meaning. Words with similar
meaning end up closer together in this vector space. Never worked with
this concept before.
Training vs Test accuracy gap confused me at first:

Training accuracy at epoch 10: 93%
Test accuracy: 85.45%

Initially thought something was wrong. Later understood this is normal
and actually expected — model performs better on data it has seen
during training vs completely unseen data. The gap here is reasonable
and not a sign of heavy overfitting.
Watching accuracy improve epoch by epoch was interesting:

Epoch 1: 51% — basically random guessing
Epoch 3: 78% — starting to understand patterns
Epoch 7: 91% — getting confident
Final test: 85.45% — real accuracy on unseen data

Dropout was another new concept — randomly switches off neurons
during training so the model doesn't just memorize the training data.
Forces it to actually learn patterns.

# Sample Predictions
All 3 test samples predicted correctly with 95-99% confidence.