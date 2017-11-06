import cv2

imgfile = 'images/model.jpg'
img = cv2.imread(imgfile)
px= img[240, 200]
print(px)

B = img.item(240, 220, 0)
G = img.item(240, 220, 1)
R = img.item(240, 220, 2)

BGR = [B, G, R]
print(BGR)

img.itemset((240,200,0), 100)

print(img.shape)
print(img.size)
print(img.dtype)

cv2.imshow('original', img)

subimg = img[200:300, 250:450]
cv2.imshow('cutting', subimg)

img[200:300, 0:200] = subimg

cv2.imshow('modified', img)

b,g,r = cv2.split(img)
cv2.imshow('b', b)
cv2.imshow('g', g)
cv2.imshow('r', r)
cv2.waitKey(0)
cv2.destroyAllWindows()