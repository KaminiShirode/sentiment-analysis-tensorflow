import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../outputs', exist_ok=True)

print("TensorFlow version:", tf.__version__)

# ── 1. LOAD DATASET ───────────────────────────────────
# IMDB dataset is built into TensorFlow — no download needed!
print("\nLoading IMDB dataset...")
NUM_WORDS = 10000
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(
    num_words=NUM_WORDS
)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Sample label (1=Positive, 0=Negative): {y_train[0]}")

# ── 2. EXPLORE DATA ───────────────────────────────────
print("\n── Data Exploration ──")
review_lengths = [len(r) for r in X_train]
print(f"Average review length : {np.mean(review_lengths):.0f} words")
print(f"Max review length     : {np.max(review_lengths)} words")
print(f"Positive reviews      : {np.sum(y_train == 1)} ({np.mean(y_train)*100:.0f}%)")
print(f"Negative reviews      : {np.sum(y_train == 0)} ({(1-np.mean(y_train))*100:.0f}%)")

# ── 3. PREPROCESS — PAD SEQUENCES ─────────────────────
# Neural networks need fixed-length input
MAX_LEN = 256
print(f"\nPadding sequences to length {MAX_LEN}...")

X_train_pad = keras.preprocessing.sequence.pad_sequences(
    X_train, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad  = keras.preprocessing.sequence.pad_sequences(
    X_test,  maxlen=MAX_LEN, padding='post', truncating='post')

print(f"X_train shape: {X_train_pad.shape}")
print(f"X_test shape : {X_test_pad.shape}")

# ── 4. BUILD NEURAL NETWORK ───────────────────────────
print("\nBuilding Neural Network...")

EMBEDDING_DIM = 32

model = keras.Sequential([
    # Embedding layer — converts word indices to dense vectors
    layers.Embedding(input_dim=NUM_WORDS,
                     output_dim=EMBEDDING_DIM,
                     input_length=MAX_LEN,
                     name='embedding'),

    # Aggregate embedded vectors
    layers.GlobalAveragePooling1D(name='pooling'),

    # Hidden layers with ReLU activation
    layers.Dense(64, activation='relu', name='hidden_1'),
    layers.Dropout(0.3, name='dropout_1'),
    layers.Dense(32, activation='relu', name='hidden_2'),
    layers.Dropout(0.3, name='dropout_2'),

    # Output layer — sigmoid for binary classification
    layers.Dense(1, activation='sigmoid', name='output')
], name='sentiment_classifier')

model.summary()

# ── 5. COMPILE MODEL ──────────────────────────────────
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ── 6. TRAIN MODEL ────────────────────────────────────
print("\nTraining model — 10 epochs...")
EPOCHS     = 10
BATCH_SIZE = 512
VAL_SPLIT  = 0.2

history = model.fit(
    X_train_pad, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    verbose=1
)

# ── 7. EVALUATE ON TEST SET ───────────────────────────
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"\n── Final Results ──")
print(f"Test Accuracy : {test_acc*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ── 8. PLOT: Accuracy and Loss Curves ─────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy', color='steelblue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy',   color='orange')
plt.title(f'Model Accuracy (Test: {test_acc*100:.1f}%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss', color='steelblue')
plt.plot(history.history['val_loss'], label='Val Loss',   color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/accuracy_loss_plot.png', dpi=150)
plt.show()
print("Saved: outputs/accuracy_loss_plot.png")

# ── 9. SAMPLE PREDICTIONS ─────────────────────────────
print("\n── Sample Predictions ──")
for i in [0, 1, 2]:
    sample    = X_test_pad[i:i+1]
    pred      = model.predict(sample, verbose=0)[0][0]
    actual    = "Positive" if y_test[i] == 1 else "Negative"
    predicted = "Positive" if pred > 0.5 else "Negative"
    confidence = pred if pred > 0.5 else 1 - pred
    print(f"\nReview {i+1}:")
    print(f"  Actual    : {actual}")
    print(f"  Predicted : {predicted} (confidence: {confidence*100:.1f}%)")
    print(f"  Correct   : {'Yes' if actual == predicted else 'No'}")

# ── 10. SAVE MODEL ────────────────────────────────────
model.save('../outputs/sentiment_model.keras')
print("\nModel saved to outputs/sentiment_model.keras")
print("\nProject complete! Check outputs/ folder for charts.")
