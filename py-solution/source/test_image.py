from PIL import Image

im = Image.open('mytwo.png')
pixel_map = im.load()
im_size = im.size
for i in range(im_size[0]):
    for j in range(im_size[1]):
        print(pixel_map[i, j])