import numpy as np
import cv2

ref_img = cv2.imread("data/ref.jpg")
tgt_img = cv2.imread("data/tgt.jpg")

ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
tgt_gray = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)

feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.01,
                       minDistance = 20.,
                       blockSize = 3,
                       useHarrisDetector = False,
                       k = 0.04)

lk_params = dict( winSize  = (15,15),
                  maxLevel = 5,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.3),
                  flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                  minEigThreshold = 1e-5)

feat_ref = cv2.goodFeaturesToTrack(ref_gray, mask = None, **feature_params)
feat_tgt = cv2.goodFeaturesToTrack(tgt_gray, mask = None, **feature_params)

feat_tgt, feat_found, feat_err = cv2.calcOpticalFlowPyrLK(ref_gray, tgt_gray, feat_ref, None, **lk_params)

for i, (ref, tgt) in enumerate(zip(feat_ref, feat_tgt)):
    print(i, " Error is ", feat_err[i], " Found is ", feat_found[i]);
    a, b = ref.ravel()
    c, d = tgt.ravel()
    cv2.line(ref_img, (a, b), (c, d), [255, 255, 255], 2)
    if (feat_err[i] < 0.8 and feat_found[i] == 1):
        cv2.circle(ref_img, (a, b), 5, [255, 0, 0], -1)
        cv2.circle(tgt_img, (c, d), 5, [255, 0, 0], -1)

p0 = (211, 432)
p1 = (211, 512)
p2 = (275, 512)
p3 = (275, 432)
ref_corners = np.array([p0, p1, p2, p3], dtype='float32')

fctrX = 0.25 * (p0[0] + p1[0] + p2[0] + p3[0]);
fctrY = 0.25 * (p0[1] + p1[1] + p2[1] + p3[1]);
fradius = 40.0

vecdistnregion = []
for i in range(len(feat_ref)):
    fdist = np.sqrt((feat_ref[i][0][0] - fctrX) * (feat_ref[i][0][0] - fctrX) + (feat_ref[i][0][1] - fctrY) * (feat_ref[i][0][1] - fctrY))
    nratio = int(fdist / fradius + 0.499999)
    vecdistnregion.append([fdist, nratio])

numfeatPtsonSticker = 0
for i in range(len(vecdistnregion)):
    if (vecdistnregion[i][1] == 1):
        numfeatPtsonSticker += 1

matchedptsref = np.array([[]])
matchedptstgt = np.array([[]])
if (numfeatPtsonSticker > 6):
    for i in range(len(feat_tgt)):
        if (feat_err[i] < 0.5 and feat_found[i] == 1 and vecdistnregion[i][1] == 1):
            refpt = feat_ref[i];
            tgtpt = feat_tgt[i];
            matchedptsref = np.append(matchedptsref, refpt, axis=1)
            matchedptstgt = np.append(matchedptstgt, tgtpt, axis=1)
else:
    for i in range(len(feat_tgt)):
        if (feat_err[i] < 0.5 and feat_found[i] == 1):
            refpt = feat_ref[i];
            tgtpt = feat_tgt[i];
            matchedptsref = np.append(matchedptsref, refpt, axis=1)
            matchedptstgt = np.append(matchedptstgt, tgtpt, axis=1)


matchedptsref = matchedptsref.reshape(-1, 2)
matchedptstgt = matchedptstgt.reshape(-1, 2)

H,_ = cv2.findHomography(matchedptsref, matchedptstgt, method=cv2.RANSAC);

ref_corners = np.array([ref_corners])
tgt_corner = cv2.perspectiveTransform(ref_corners, H);
tgt_corner = tgt_corner.reshape(-1, 2)

cv2.line(ref_img, p0, p1, [255, 0, 0], 2);
cv2.line(ref_img, p1, p2, [255, 0, 0], 2);
cv2.line(ref_img, p2, p3, [255, 0, 0], 2);
cv2.line(ref_img, p3, p0, [255, 0, 0], 2);

tp0 = (tgt_corner[0][0], tgt_corner[0][1]);
tp1 = (tgt_corner[1][0], tgt_corner[1][1]);
tp2 = (tgt_corner[2][0], tgt_corner[2][1]);
tp3 = (tgt_corner[3][0], tgt_corner[3][1]);

cv2.line(tgt_img, tp0, tp1, [255, 0, 0], 2);
cv2.line(tgt_img, tp1, tp2, [255, 0, 0], 2);
cv2.line(tgt_img, tp2, tp3, [255, 0, 0], 2);
cv2.line(tgt_img, tp3, tp0, [255, 0, 0], 2);

cv2.imshow("ref", ref_img)
cv2.imshow("tgt", tgt_img)

cv2.waitKey(0)
cv2.destroyAllWindows()