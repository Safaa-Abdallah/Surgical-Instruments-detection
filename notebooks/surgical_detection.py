"""
Surgical Instruments Detection - CNN Model
Computer Vision Project for Operating Room Equipment Classification
Author: Safaa Kamaleldin
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

def create_cnn_model():
"""
Create CNN model based on the project report description
Input: 128x128 RGB images
Output: 3 classes (Operation Table, Surgery Light, Operation Room)
"""
model = Sequential([
# First Convolutional Layer - 32 filters
Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
MaxPooling2D(2, 2),

# Second Convolutional Layer - 64 filters
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2, 2),

# Flatten and Fully Connected Layers
Flatten(),
Dense(128, activation='relu'),
Dense(3, activation='softmax') # 3 output classes
])

return model

def main():
"""
Main function to demonstrate the surgical instruments detection model
"""
print("🏥 Surgical Instruments Detection Model")
print("=" * 50)

# Create model
model = create_cnn_model()

# Compile model
model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)

# Display model summary
print("📊 Model Architecture:")
model.summary()

print("\n✅ Model Created Successfully!")
print("🎯 Expected Accuracy: 72% (as reported in project)")
print("📈 Classes: Operation Table, Surgery Light, Operation Room")
print("🖼️ Input Size: 128x128 RGB images")

# Model configuration
print("\n🔧 Model Configuration:")
print(f"- Total Layers: {len(model.layers)}")
print(f"- Total Parameters: {model.count_params():,}")
print(f"- Output Shape: {model.output_shape}")

if __name__ == "__main__":
main()
