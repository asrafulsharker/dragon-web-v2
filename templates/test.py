# import tensorflow as tf
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the fine-tuned model
# fine_tuned_model_path = '/kaggle/working/model.h5'
# fine_tuned_model = tf.keras.models.load_model(fine_tuned_model_path)

# # Load the pre-trained EfficientNetB0 model
# model = EfficientNetB0(weights='imagenet', include_top=True)

# # Choose the last convolutional layer name
# last_conv_layer_name = 'top_activation'

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     # First, we create a model that maps the input image to the activations
#     # of the last conv layer as well as the output predictions
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     # Preprocess the image for EfficientNetB0
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

#     # Then, we compute the gradient of the top predicted class for our input image
#     # with respect to the activations of the last conv layer
#     with tf.GradientTape() as tape:
#         last_conv_layer_output, preds = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]

#     # This is the gradient of the output neuron (top predicted or chosen)
#     # with regard to the output feature map of the last conv layer
#     grads = tape.gradient(class_channel, last_conv_layer_output)

#     # This is a vector where each entry is the mean intensity of the gradient
#     # over a specific feature map channel
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     # We multiply each channel in the feature map array
#     # by "how important this channel is" with regard to the top predicted class
#     # then sum all the channels to obtain the heatmap class activation
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     # For visualization purposes, normalize the heatmap between 0 & 1
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()

# def guided_backpropagation(img_array, model, target_size=(224, 224)):
#     # Create a model that computes the guided backpropagation
#     gb_model = tf.keras.models.Model(model.inputs, [model.layers[-3].output])

#     # Preprocess the image for EfficientNetB0
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

#     # Compute the guided backpropagation
#     with tf.GradientTape() as tape:
#         inputs = tf.cast(img_array, tf.float32)
#         tape.watch(inputs)
#         outputs = gb_model(inputs)

#     # Get the gradients of the outputs with respect to the inputs
#     grads = tape.gradient(outputs, inputs)[0]

#     # Resize the gradients to match the size of the target heatmap
#     grads = tf.image.resize(grads, target_size)

#     return grads.numpy()

# # Load an example image
# img_path = '/kaggle/input/leaf-2/data2/Tomato_mosaic_virus/Tomato_mosaic_virus0001.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# img_array = image.img_to_array(img)[tf.newaxis, ...]

# # Generate Grad-CAM heatmap
# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# # Generate guided backpropagation
# guided_bp = guided_backpropagation(img_array, model, target_size=heatmap.shape[:2])

# # Guided Grad-CAM++
# guided_gradcam = np.multiply(guided_bp, heatmap[..., np.newaxis])
# guided_gradcam = tf.maximum(guided_gradcam, 0) / np.max(guided_gradcam)

# # Superimpose the heatmap on the original image
# img = image.img_to_array(img)
# heatmap = np.uint8(255 * heatmap)
# jet = plt.get_cmap("jet")
# jet_colors = jet(np.arange(256))[:, :3]
# jet_heatmap = jet_colors[heatmap]
# jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
# jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
# jet_heatmap = image.img_to_array(jet_heatmap)

# # Convert the heatmap to RGB
# heatmap_colormap = plt.cm.plasma(heatmap)
# heatmap_colormap = (heatmap_colormap[:, :, :3] * 255).astype(np.uint8)

# # Resize heatmap_colormap to match the original image size
# heatmap_colormap = cv2.resize(heatmap_colormap, (img_array.shape[2], img_array.shape[1]))

# # Convert the PIL image to a NumPy array
# superimposed_img = cv2.addWeighted(np.array(img).astype('uint8'), 0.6, heatmap_colormap, 0.4, 0)

# # Display the original image, Grad-CAM heatmap, and Guided Grad-CAM++ in a row
# plt.figure(figsize=(15, 5))

# # Original Image
# plt.subplot(1, 3, 1)
# plt.imshow(img.astype('uint8'))
# plt.title('Original Image')

# # Grad-CAM Heatmap
# plt.subplot(1, 3, 2)
# plt.imshow(superimposed_img_array)
# plt.title('Grad-CAM Heatmap')

# # Guided Grad-CAM++
# plt.subplot(1, 3, 3)
# plt.imshow(superimposed_img)
# plt.title('Guided Grad-CAM++')

# plt.show()
