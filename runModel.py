import tensorflow as tf
import numpy
from PIL import Image
import glob
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def categorical_to_mask(im):
    mask = tf.dtypes.cast(tf.argmax(im, axis=2), 'float32') / 255.0
    return mask

def load_files(path, target_size, scale_factor):    
    image_list = []
    filenames = glob.glob(path)
    filenames.sort()
    for filename in filenames:
        im = Image.open(filename)
        w, h = im.size
        im = im.resize((target_size, target_size))
        im=numpy.asarray(im) / scale_factor
        image_list.append(im)
    return numpy.asarray(image_list)

semantic_segmentation_model = tf.keras.models.load_model('C:\\Users\\User\\Documents\\University\\Year_4\\Semester_2\\EGH400-2\\maritime-collision-avoidance-thesis\\Semantic_Segmentation\\saved_model\\semantic_segmentation_model', compile=False)

test_image = load_files('C:\\Users\\User\\Documents\\University\\Year_4\\Semester_2\\EGH400-2\\maritime-collision-avoidance-thesis\\Semantic_Segmentation\\testImage.jpg', 128, 255.0)

predict = semantic_segmentation_model.predict(test_image)
output = categorical_to_mask(predict[0,:,:,:])

imgplot = plt.imshow(output)
plt.show()
