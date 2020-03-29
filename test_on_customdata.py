import numpy as np
from resnetimp import Resnet
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions

model = Resnet(include_top=True,weights='imagenet')
model.summary()
model.layers[-1].get_config()
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
#
preds = model.predict(x)
print('Predicted:', decode_predictions(preds))