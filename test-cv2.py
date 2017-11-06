import cv2
import matplotlib.pyplot as plt


def showImage():
    imgfile = 'images/model.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([])
    plt.yticks([])
    plt.title('model')
    plt.show()

    k = cv2.waitKey(0) & 0xFF

    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('c'):
        cv2.imwrite('images/model_copy.jpg', img)
        cv2.destroyAllWindows()

showImage()
