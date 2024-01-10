import numpy
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
APP_ROOT= os.path.dirname(os.path.abspath(__file__))

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.5):
    # Load the original image

    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]



    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)


    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return np.array(superimposed_img)




def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    # --------------------------------------------------------------------------------

    grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer_name).output,model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output,preds= grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]


    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    # grads = tape.gradient(class_channel1, convOutputs5)
    grads = tape.gradient(class_channel, last_conv_layer_output)
    #
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0 ,1, 2))

    # We multiply each channel in the feature map array
    # by "how important this
    # channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1

    cam = heatmap / np.max(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def heap(p1,newName,user_id):
    plt.clf()
    plt.cla()
    plt.close()
    target = os.path.join(APP_ROOT, 'static/Patient_images')
    target = "/".join([target, p1])
    img_path = target

    classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
               'Pleural_Thickening', 'No Finding']

    my_model = tf.keras.models.load_model('static/models/heap/best-vgg16-0402-model.h5')
    # my_model.load_weights('m2-23.h5')
    my_model.summary()


    dim = (224, 224, 3)
    X = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    X[0, :] = cv2.resize(cv2.imread(img_path, 1), dim[:2])
    predictions = my_model.predict(X)
    prediction_indexes = np.argmax(predictions, axis=1)


    predictions_sorted, classes_sorted = zip(*sorted(zip(predictions[0], classes), reverse=True))

    print("Prediction:\n")
    for disease in classes_sorted:
        print(disease, ": ", "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100), "%")


    image = cv2.imread(img_path, 1)
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    # image=load_image(image)
    print(image.shape)

    preds = my_model.predict(image)
    i = np.argmax(predictions[0])

    heatmap = make_gradcam_heatmap(image, my_model, 'block5_conv3')

    # Display heatmap
    fig = save_and_display_gradcam(img_path, heatmap,cam_path='static/Patient_images/User2/khattak.jpg')
    if not os.path.exists('static/heapmap/User' + str(user_id)):
        os.makedirs('static/heapmap/User' + str(user_id))
    cv2.imwrite('static/heapmap/User' + str(user_id) + '/' + newName + '.jpg', fig)

