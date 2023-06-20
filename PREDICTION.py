import tensorflow as tf 
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.utils import load_img
img_width, img_height = 64, 64

model = tf.keras.models.load_model('defects_classification_model.h5')
test_image = tf.keras.utils.load_img('klarf/DP57572.1Y_09.tif', target_size = (img_width, img_height))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = test_image/255
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
print (result)

if result[0][0] > result[0][1] and result[0][0] > result[0][2] and result[0][0] > result[0][3] and result[0][0] > result[0][4]:
    print('Contamination particle')
elif result[0][1] > result[0][0] and result[0][1] > result[0][2] and result[0][1] > result[0][3] and result[0][1] > result[0][4]:
    print('Pattern defect')
elif result[0][2] > result[0][0] and result[0][2] > result[0][1] and result[0][2] > result[0][3] and result[0][2] > result[0][4]:
    print('Probe mark')
elif result[0][3] > result[0][0] and result[0][3] > result[0][1] and result[0][3] > result[0][2] and result[0][3] > result[0][4]:
    print('Scratches') 
elif result[0][4] > result[0][0] and result[0][4] > result[0][1] and result[0][4] > result[0][2] and result[0][4] > result[0][3]:
    print('Others')