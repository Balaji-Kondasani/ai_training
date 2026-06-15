import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers

# --- 1. Custom Layer Subclassing ---
class CustomDenseLayer(layers.Layer):
    """
    Custom Dense Layer implementing w * x + b
    """
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        # build is called once when the layer shape is first resolved
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="w"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="b"
        )
        
    def call(self, inputs):
        # Performs the mathematical forward pass computation
        return tf.matmul(inputs, self.w) + self.b

# --- 2. Custom Loss Function ---
def custom_huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber Loss: Less sensitive to outliers than MSE, but smoother than MAE near zero.
    """
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, squared_loss, linear_loss)

def main():
    print("=== Day 5: Custom Keras Layers & Loops ===")
    
    # 1. Test Custom Layer
    print("Testing CustomDenseLayer:")
    custom_dense = CustomDenseLayer(units=4)
    x_test = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    output = custom_dense(x_test)
    print(f"  Input Shape:  {x_test.shape}")
    print(f"  Output Shape: {output.shape}")
    print(f"  Weights Variable: {custom_dense.w.shape}")
    print(f"  Output value: {output.numpy()}\n")
    
    # 2. Test Custom Loss Function
    y_t = tf.constant([2.0, 5.0])
    y_p = tf.constant([1.8, 10.0])  # Sample 2 is an outlier (error = 5.0)
    loss_vals = custom_huber_loss(y_t, y_p)
    print(f"Custom Huber Loss values for predictions {y_p.numpy()}: {loss_vals.numpy()}\n")
    
    # 3. Custom Training Loop (GradientTape)
    print("--- Running Custom Training Loop ---")
    np_X = np = tf.random.normal((20, 3))
    np_y = tf.random.normal((20, 1))
    
    # Simple model
    model = tf.keras.Sequential([
        CustomDenseLayer(units=1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    
    epochs = 5
    for epoch in range(1, epochs + 1):
        with tf.GradientTape() as tape:
            # 1. Forward pass
            predictions = model(np_X)
            # 2. Calculate custom loss
            loss_value = tf.reduce_mean(custom_huber_loss(np_y, predictions))
            
        # 3. Compute gradients of loss with respect to trainable variables
        gradients = tape.gradient(loss_value, model.trainable_variables)
        
        # 4. Apply gradients using optimizer
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f"  Epoch {epoch} | Loss: {loss_value.numpy():.6f}")

if __name__ == "__main__":
    main()
