import tensorflow as tf

# Define model save path
model_save_path = "./mobilenetv2_saved_model"

# Load a pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add classification layers
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')  # Adjust number of classes as needed
])

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Save the model in TensorFlow SavedModel format
model.save(model_save_path)

print(f"Model saved at {model_save_path}")
