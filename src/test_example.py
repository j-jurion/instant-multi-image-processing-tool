from example import process_image
from imip import imip

imip.debug_fn(process_image, "./images")
print(len(imip.debug_images))
imip.show_debug_images()
