import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, ReLU, DepthwiseConv2D, Conv2DTranspose, Concatenate, Softmax
from tensorflow.keras.losses import MeanSquaredError, binary_crossentropy

# Define the berhu_loss function
def berhu_loss(y_true, y_pred, c=0.2):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    diff = tf.abs(y_true - y_pred)
    huber_loss = tf.where(diff < c, 0.5 * tf.square(diff), c * diff - 0.5 * c ** 2)
    berhu_loss = tf.where(diff < c, huber_loss, tf.square(diff) / (2 * c))
    return tf.reduce_mean(berhu_loss)
# Define the SSIM loss function
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Define the combined loss function
def combined_loss1(y_true, y_pred):
    mse = MeanSquaredError()(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    return mse + ssim

# Define the SSIM loss function
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Define the combined loss function
def combined_loss2(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + ssim_loss(y_true, y_pred)
# Load the saved models
model1 = keras.models.load_model('/content/drive/My Drive/Mobile_net_subpixel3.h5', custom_objects={'berhu_loss': berhu_loss})
model2 = keras.models.load_model('/content/drive/My Drive/Mobile_net_attention2.h5', custom_objects={'berhu_loss': berhu_loss})
model3 = keras.models.load_model('/content/drive/My Drive/denseunet_with_attention.h5', custom_objects={'combined_loss2': combined_loss2})
model4 = keras.models.load_model('/content/drive/My Drive/resunet_with_attention.h5', custom_objects={'combined_loss2': combined_loss2})
model5 = keras.models.load_model('/content/drive/My Drive/unet_with_batchnorm.h5', custom_objects={'combined_loss1': combined_loss1})


# Define the input image size
input_size = (224, 224)

# Function to preprocess the input image
def preprocess_image(image):
    resized_image = cv2.resize(image, input_size)
    normalized_image = (resized_image.astype(np.float32) - resized_image.min()) / (resized_image.max() - resized_image.min())
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Function to postprocess the predicted depth map
def postprocess_depth(depth_map):
    depth_map = np.squeeze(depth_map, axis=0)
    return depth_map

# Function to display depth map with reverse plasma colormap
def display_depth_map(depth_map, title):
    # Normalize the depth map for visualization
    normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Define the colormap
    cmap = "plasma_r"

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Display the depth map with the reverse plasma colormap
    im = ax.imshow(normalized_depth, cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    # Disable axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    return fig

# Function to calculate depth score
def calculate_depth_score(predicted_depth, ground_truth_depth):
    # Remove the batch size dimension from the predicted depth map
    predicted_depth = np.squeeze(predicted_depth, axis=0)

    # Convert both depth maps to float32 data type
    predicted_depth = predicted_depth.astype(np.float32)
    ground_truth_depth = ground_truth_depth.astype(np.float32)

    # Resize the ground truth depth map to match the predicted depth map resolution
    ground_truth_depth = cv2.resize(ground_truth_depth, (predicted_depth.shape[1], predicted_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Calculate the depth score using berhu loss
    depth_score = berhu_loss(ground_truth_depth, predicted_depth)
    return depth_score

# Streamlit app
def app():
    st.title("Depth Map Prediction")
    uploaded_image = st.file_uploader("Choose an input image", type=["jpg", "png", "jpeg"])
    uploaded_ground_truth = st.file_uploader("Choose a ground truth depth map (optional)", type=["jpg", "png", "jpeg"])

    if st.button("Predict"):
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_image = preprocess_image(image)

            # Predict the depth maps
            predicted_depth1 = model1.predict(input_image)
            predicted_depth2 = model2.predict(input_image)
            predicted_depth3 = model3.predict(input_image)
            predicted_depth4 = model4.predict(input_image)
            predicted_depth5 = model5.predict(input_image)


            # Postprocess the predicted depth maps
            depth_map_1 = postprocess_depth(predicted_depth1)
            depth_map_2 = postprocess_depth(predicted_depth2)
            depth_map_3 = postprocess_depth(predicted_depth3)
            depth_map_4 = postprocess_depth(predicted_depth4)
            depth_map_5 = postprocess_depth(predicted_depth5)

            # Display the input image and predicted depth maps
            st.image(image, caption="Input Image", use_column_width=True)
            st.pyplot(display_depth_map(depth_map_1, "Predicted Depth Map (MobileNetV2)"))
            st.pyplot(display_depth_map(depth_map_2, "Predicted Depth Map (DenseNet121)"))
            st.pyplot(display_depth_map(depth_map_3, "Predicted Depth Map (Dense_u-net_with_attention)"))
            st.pyplot(display_depth_map(depth_map_4, "Predicted Depth Map (Res_u-net_with_attention)"))
            st.pyplot(display_depth_map(depth_map_5, "Predicted Depth Map (U-net)"))

            # Calculate and display the depth scores if ground truth is provided
            if uploaded_ground_truth is not None:
                file_bytes = np.asarray(bytearray(uploaded_ground_truth.read()), dtype=np.uint8)
                ground_truth_depth = cv2.imdecode(file_bytes, 0)  # Load as grayscale

                depth_score1 = calculate_depth_score(predicted_depth1, ground_truth_depth)
                depth_score2 = calculate_depth_score(predicted_depth2, ground_truth_depth)
                st.title("Depth Score (Higher is Better)")
                st.write(f"(MobileNetV2): {depth_score1:.4f}")
                st.write(f"(DenseNet121): {depth_score2:.4f}")
                st.write(f"(Dense_u-net_with_attention): {depth_score1:.4f}")
                st.write(f"(Res_u-net_with_attention): {depth_score2:.4f}")
                st.write(f"(U-net): {depth_score1:.4f}")
        else:
            st.warning("Please upload an input image to predict the depth map.")

if __name__ == "__main__":
    app()