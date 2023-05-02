from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from stardist.matching import matching

# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

img, y_true = test_image_nuclei_2d(return_mask=True)

y_pred, _ = model.predict_instances(normalize(img))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1, 2, 2)
plt.imshow(render_label(y_pred, img=img))
plt.axis("off")
plt.title("prediction + input overlay")

metrics = matching(y_true, y_pred)

print(metrics)
