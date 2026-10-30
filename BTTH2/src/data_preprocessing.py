import idx2numpy
import cv2 as cv

images = idx2numpy.convert_from_file("data/train-images-idx3-ubyte")
labels = idx2numpy.convert_from_file("data/train-labels-idx1-ubyte")

image = images[0]
label = labels[0]

print(f"Kich thuoc tap train: {images.shape}")
print(f"Nhan cua anh dau tien: {label}")

cv.imshow("Show digital number", image)
if cv.waitKey() == ord("q"):
    cv.destroyAllWindows()