import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib

# Oluşturulacak dosyaların kaydedileceği klasör
results_dir = 'results'
models_dir = 'models'

# Gerekli klasörlerin varlığını kontrol et, yoksa oluştur
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load data
df = pd.read_csv('data/cleaned_dataset.csv')

# Categorize injury durations
def categorize_injury(days):
    if days <= 7:  # 0-7 days
        return 1   # Minimal injury
    elif days <= 28:
        return 2   # Mild/Moderate injury
    elif days <= 84:
        return 3   # Severe injury
    else:
        return 4   # Long-term injury

# Recategorize target variable
if 'season_days_injured' in df.columns:
    df['injury_category'] = df['season_days_injured'].apply(categorize_injury)
else:
    raise KeyError("The column 'season_days_injured' does not exist in the dataset.")

# Define features and target
features = [
    # Demographic and Physical Features
    'age',
    'bmi',
    'work_rate_numeric',

    # FIFA Features
    'fifa_rating',
    'pace',
    'physic',

    # Game Statistics
    'season_minutes_played',
    'season_matches_in_squad',
    'minutes_per_game_prev_seasons',
    'avg_games_per_season_prev_seasons',

    # Injury History
    'avg_days_injured_prev_seasons',
    'cumulative_days_injured'
]

X = df[features]
y = df['injury_category']

# Encode target variable if necessary
# Assuming injury_category is already numerical (1-4)
# If it's categorical, uncomment the following lines:
# le = LabelEncoder()
# y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Feature scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Convert targets to categorical
num_classes = y.nunique()
y_train_cat = to_categorical(y_train_res - 1, num_classes=num_classes)
y_test_cat = to_categorical(y_test - 1, num_classes=num_classes)

# Save scaler for future use
scaler_path = os.path.join(models_dir, 'robust_scaler.pkl')
joblib.dump(scaler, scaler_path)

# Build the Neural Network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_scaled,
    y_train_cat,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0  # Eğitim sırasında herhangi bir çıktı göstermemek için verbose=0
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)

# Predictions
y_pred_prob = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1) + 1  # Adding 1 to match original labels

# Metrics
accuracy_val = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Classification Report
report = classification_report(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Kaydetme işlemleri
with open(os.path.join(results_dir, 'model_metrics.txt'), 'w') as f:
    f.write(f'Neural Network Test Accuracy: {accuracy_val:.4f}\n')
    f.write(f'Precision: {precision:.4f}\n')
    f.write(f'Recall: {recall:.4f}\n')
    f.write(f'F1 Score: {f1:.4f}\n\n')
    f.write("Classification Report:\n")
    f.write(report)

# Confusion Matrix'i kaydet
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(1, num_classes+1), 
            yticklabels=range(1, num_classes+1))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
confusion_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path)
plt.close()

# Plot Training History
# Accuracy Plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
accuracy_plot_path = os.path.join(results_dir, 'training_accuracy.png')
plt.savefig(accuracy_plot_path)
plt.close()

# Loss Plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
loss_plot_path = os.path.join(results_dir, 'training_loss.png')
plt.savefig(loss_plot_path)
plt.close()

# Save the model
model_path = os.path.join(models_dir, 'neural_network_model.h5')
model.save(model_path)

# Ayrıca eğitim geçmişini kaydetmek isterseniz:
import json
history_path = os.path.join(results_dir, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history.history, f)