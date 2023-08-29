import os
import pydicom
from PIL import Image

input_directory, output_directory= '/home/ec2-user/rsna/stage_2_train_images', '/home/ec2-user/rsna/stage_2_train_images_png'
for filename in os.listdir(input_directory):
    if filename.endswith('.dcm'):
        dataset = pydicom.dcmread(os.path.join(input_directory, filename))
        pixel_array_numpy = dataset.pixel_array
        image = Image.fromarray(pixel_array_numpy)
        png_filename = os.path.splitext(filename)[0] + '.png'
        image.save(os.path.join(output_directory, png_filename))
input_directory, output_directory= '/home/ec2-user/rsna/stage_2_test_images', '/home/ec2-user/rsna/stage_2_test_images_png'
for filename in os.listdir(input_directory):
    if filename.endswith('.dcm'):
        dataset = pydicom.dcmread(os.path.join(input_directory, filename))
        pixel_array_numpy = dataset.pixel_array
        image = Image.fromarray(pixel_array_numpy)
        png_filename = os.path.splitext(filename)[0] + '.png'
        image.save(os.path.join(output_directory, png_filename))
