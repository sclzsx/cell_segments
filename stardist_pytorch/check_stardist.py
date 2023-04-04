import sys
from matplotlib import pyplot as plt
from stardist.models import StarDist2D
from collections import Counter

# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

###################################################
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt

img = test_image_nuclei_2d()

labels, _ = model.predict_instances(normalize(img))
print(img.shape, labels.shape)
print(Counter(labels.flatten()))
# plt.imshow(img)
plt.show()
sys.exit()

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=img))
plt.axis("off")
plt.title("prediction + input overlay")

# plt.show()

###################################

# create some example ground-truth and dummy prediction data
from stardist.data import test_image_nuclei_2d
from scipy.ndimage import rotate
_, y_true = test_image_nuclei_2d(return_mask=True)
y_pred = rotate(y_true, 2, order=0, reshape=False)

# compute metrics between ground-truth and prediction
from stardist.matching import matching

metrics =  matching(y_true, y_pred)

print(metrics)

#########################################3

from stardist.matching import matching_dataset

metrics = matching_dataset([y_true, y_true], [y_pred, y_pred])

print(metrics)