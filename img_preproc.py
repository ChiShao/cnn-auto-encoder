from PIL import Image
import os

file_name = "pole.jpg"
unified_scale = 256, 256
img = Image.open(file_name).resize(unified_scale, Image.LANCZOS).save("small_%s" % file_name)
