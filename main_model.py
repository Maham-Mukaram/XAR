# load and evaluate a saved model
import os
import numpy as np
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB4
import efficientnet.tfkeras as efn
import tensorflow.compat.v1 as tf1
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import pandas as pd
tf1.disable_v2_behavior()
import cv2
APP_ROOT= os.path.dirname(os.path.abspath(__file__))

def model_working(adder):
    classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
               'Pleural_Thickening', 'No Finding']
    num_gpus = tf1.config.list_physical_devices('GPU')
    cross_device_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=len(num_gpus))
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
    with strategy.scope():
        my_model = tf.keras.models.load_model(
            r'static/models/m2-23_model.h5')
        my_model.load_weights(r'static/models/m2-23.h5')

    dim = (380, 380, 3)
    X = np.zeros((1, 380, 380, 3), dtype=np.uint8)
    X[0, :] = cv2.resize(cv2.imread(adder, 1),
                         dim[:2])
    predictions = my_model.predict(X)

    predictions_sorted, classes_sorted = zip(*sorted(zip(predictions[0], classes), reverse=True))

    return classes_sorted, predictions_sorted

def predict(p1, newName, user_id):
    plt.clf()
    plt.cla()
    plt.close()
    target = os.path.join(APP_ROOT, 'static/Patient_images')
    target = "/".join([target, p1])
    classes_sorted, predictions_sorted = model_working(target)
    Atelectasis, Cardiomegaly, Effusion, Emphysema, Infiltration, Mass, Nodule, Pneumothorax, Consolidation, Edema, Pleural_Thickening, No_Finding = '', '', '', '', '', '', '', '', '', '', '', ''
    for disease in classes_sorted:
        if disease == 'Atelectasis':
            Atelectasis = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
        if disease == 'Cardiomegaly':
            Cardiomegaly = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
        if disease == 'Effusion':
            Effusion = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
        if disease == 'Infiltration':
            Infiltration = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
        if disease == 'Mass':
            Mass = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
        if disease == 'Nodule':
            Nodule = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
        if disease == 'Pneumothorax':
            Pneumothorax = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
        if disease == 'Consolidation':
            Consolidation = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
        if disease == 'Edema':
            Edema = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
        if disease == 'Emphysema':
            Emphysema = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
        if disease == 'Pleural_Thickening':
            Pleural_Thickening = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
        if disease == 'No Finding':
            No_Finding = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(disease)]) * 100)
    finding = ''
    maxi = 0
    for d in classes_sorted:
        if float("{:.2f}".format(float(predictions_sorted[classes_sorted.index(d)]) * 100)) >= float(maxi):
            maxi = "{:.2f}".format(float(predictions_sorted[classes_sorted.index(d)]) * 100)
            finding = d
    if not os.path.exists('static/classification/User' + str(user_id)):
        os.makedirs('static/classification/User' + str(user_id))
    path = os.path.join(APP_ROOT, 'static/classification/User'+ str(user_id)+'/' + newName + '.jpg')
    data = {'Probability': [Atelectasis, Cardiomegaly, Effusion, Emphysema, Infiltration, Mass, Nodule, Pneumothorax, Consolidation, Edema, Pleural_Thickening, No_Finding]}
    df = pd.DataFrame(data, index=['Atelectasis', 'Cardiomegaly', 'Effusion', 'Emphysema', 'Infiltration', 'Mass', 'Nodule', 'Pneumothorax', 'Consolidation', 'Edema', 'Pleural_Thickening', 'No_Finding'])
    df = df.astype(float)
    df.plot.barh(rot=0)
    plt.ylabel('Disease')
    plt.xlabel('Probability')
    plt.savefig(path, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    path1= 'static/classification/User'+ str(user_id)+'/' + newName + '.jpg'
    return finding, path1

