import numpy as np
import PIL.Image as Image

def load_image(target_dir):
    image = Image.open(target_dir)
    short_edge = min(image.size)
    image = image.crop(box=(0,0,short_edge,short_edge))
    print(image.size)
    image = image.resize((224,224))
    image = np.asarray(image)/255
    return image

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


# def load_image2(path, height=None, width=None):
#     # load image
    
#     img = skimage.io.imread(path)
#     img = img / 255.0
#     if height is not None and width is not None:
#         ny = height
#         nx = width
#     elif height is not None:
#         ny = height
#         nx = img.shape[1] * ny / img.shape[0]
#     elif width is not None:
#         nx = width
#         ny = img.shape[0] * nx / img.shape[1]
#     else:
#         ny = img.shape[0]
#         nx = img.shape[1]
#     return skimage.transform.resize(img, (ny, nx))


# def test():
#     img = skimage.io.imread("./test_data/starry_night.jpg")
#     ny = 300
#     nx = img.shape[1] * ny / img.shape[0]
#     img = skimage.transform.resize(img, (ny, nx))
#     skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
