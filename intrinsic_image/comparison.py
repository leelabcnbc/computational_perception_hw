import numpy as np
import os
import sys
import skimage.transform
import myhtml
import intrinsic
from skimage import color
import imageio
import matplotlib.pyplot as plt

# The following objects were used in the evaluation. For the learning algorithms
# (not included here), we used two-fold cross-validation with the following
# randomly chosen split.


SET1 = ['box', 'cup1', 'cup2', 'dinosaur', 'panther', 'squirrel', 'sun', 'teabag2']
SET2 = ['deer', 'frog1', 'frog2', 'paper1', 'paper2', 'raccoon', 'teabag1', 'turtle']
ALL_TAGS = SET1 + SET2

# ALL_TAGS = ['teabag1']
# The following four objects weren't used in the evaluation because they have
# slight problems, but you may still find them useful.
EXTRA_TAGS = ['apple', 'pear', 'phone', 'potato']

# Use L1 to compute the final results. (For efficiency, the parameters are still
# tuned using least squares.)
USE_L1 = False

# Output of the algorithms will be saved here
if USE_L1:
    RESULTS_DIR = 'results_L1'
else:
    RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)

def print_dot(i, num):
    NEWLINE_EVERY = 50
    sys.stdout.write('.')
    if (i+1) % NEWLINE_EVERY == 0:
        sys.stdout.write('  [%d/%d]' % (i+1, num))
    if (i+1) % NEWLINE_EVERY == 0 or i+1 == num:
        sys.stdout.write('\n')
    sys.stdout.flush()
        
def save_estimates(gen, image, est_shading, est_refl, mask):
    """Outputs the estimated shading and reflectance images to an HTML
    file. Does nothing if Python Imaging Library is not installed."""
    image = image / np.max(image)
    est_shading = est_shading / np.max(est_shading)
    est_refl = est_refl / np.max(est_refl)
    # gamma correct
    image = np.where(mask, image ** (1./2.2), 1.)
    est_shading = np.where(mask, est_shading ** (1./2.2), 1.)
    est_refl = np.where(mask, est_refl ** (1./2.2), 1.)
    output = np.concatenate([image, est_shading, est_refl], axis=1)
    gen.image(output)


def run_experiment():
    """Script for running the algorithmic comparisons from the paper

        Roger Grosse, Micah Johnson, Edward Adelson, and William Freeman,
          Ground truth dataset and baseline evaluations for intrinsic
          image algorithms.

    Evaluates each of the algorithms on the MIT Intrinsic Images dataset
    with hold-one-out cross-validation.
    
    For each algorithm, it precomputes the error scores for all objects with
    all parameter settings. Then, for each object, it chooses the parameters
    which achieved the smallest average error on the other objects. The
    results are all output to the HTML file results/index.html."""
    
    assert os.path.isdir(RESULTS_DIR), '%s: directory does not exist' % RESULTS_DIR
    
    '''
    estimators = [('Baseline (BAS)', intrinsic.BaselineEstimator),
                  ('Grayscale Retinex (GR-RET)', intrinsic.GrayscaleRetinexEstimator),
                  ('Color Retinex (COL-RET)', intrinsic.ColorRetinexEstimator),
                  ("Weiss's Algorithm (W)", intrinsic.WeissEstimator),
                  ('Weiss + Retinex (W+RET)', intrinsic.WeissRetinexEstimator)]
                  '''
    estimators = [('Color Retinex (COL-RET)', intrinsic.ColorRetinexEstimator)]

    gen = myhtml.Generator('Intrinsic image results', RESULTS_DIR)
    for tag in ALL_TAGS:
        # tag = ALL_TAGS
        ntags = len(tag)
        
        for e, (name, EstimatorClass) in enumerate(estimators):
            inp = EstimatorClass.get_input(tag)
            inp = inp + (USE_L1,)
            choices = EstimatorClass.param_choices()
            image = intrinsic.load_object(tag, 'diffuse')
        
            image = color.gray2rgb(image)

            image = np.mean(image, axis=2)
            mask = intrinsic.load_object(tag, 'mask')
            mask = imageio.imread(os.path.join("data", tag, "mask.png"))
            mask = color.rgb2gray(mask)

            params = choices[0]
            estimator = EstimatorClass(**params)
            est_shading, est_refl = estimator.estimate_shading_refl(*inp)

            save_estimates(gen, image, est_shading, est_refl, mask)
    
if __name__ == '__main__':
    run_experiment()
