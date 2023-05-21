"""docstring"""
import os
import pydicom
from PIL import Image

INPUTDIR = '/Users/taeyeonpaik/Downloads/rsna-pneumonia-detection-challenge/stage_2_train_images'
OUTDIR = '/Users/taeyeonpaik/Downloads/rsna-pneumonia-detection-challenge/stage_2_train_images_png'

for filename in os.listdir(INPUTDIR):
    if filename.endswith('.dcm'):
        ds = pydicom.dcmread(os.path.join(INPUTDIR, filename))
        pixel_array_numpy = ds.pixel_array
        image = Image.fromarray(pixel_array_numpy)
        png_filename = os.path.splitext(filename)[0] + '.png'
        image.save(os.path.join(OUTDIR, png_filename))

INPUTDIR = '/Users/taeyeonpaik/Downloads/rsna-pneumonia-detection-challenge/stage_2_test_images'
OUTDIR = '/Users/taeyeonpaik/Downloads/rsna-pneumonia-detection-challenge/stage_2_test_images_png'

for filename in os.listdir(INPUTDIR):
    if filename.endswith('.dcm'):
        ds = pydicom.dcmread(os.path.join(INPUTDIR, filename))
        pixel_array_numpy = ds.pixel_array
        image = Image.fromarray(pixel_array_numpy)
        png_filename = os.path.splitext(filename)[0] + '.png'
        image.save(os.path.join(OUTDIR, png_filename))
