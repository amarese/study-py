import numpy as np
import cv2
import math


def testOpticalFlow():
    cpath = "data/"
    refname = cpath + "ref.jpg"
    tgtname = cpath + "tgt.jpg"
    regImg = cv2.imread(refname)
    regImggray = cv2.cvtColor(regImg, cv2.COLOR_BGR2GRAY)
    tgtImg = cv2.imread(tgtname)
    tgtImggray = cv2.cvtColor(tgtImg, cv2.COLOR_BGR2GRAY)

    img_sz = tgtImg.size
    imgALL = (img_sz, 1)

    feature_params = dict(maxCorners=50,
                          qualityLevel=0.01,
                          minDistance=20,
                          blockSize=3,
                          useHarrisDetector=False,
                          k=0.04)

    # get good features for
    feat_ref = cv2.goodFeaturesToTrack(regImggray, mask=None, **feature_params)
    feat_tgt = cv2.goodFeaturesToTrack(tgtImggray, mask=None, **feature_params)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.3),
                     flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                     minEigThreshold=1e-5)
    feat_tgt, feat_found, feat_err = cv2.calcOpticalFlowPyrLK(regImggray, tgtImggray, feat_ref, feat_tgt, **lk_params)

    # draw the tracks
    for i, (ref, tgt, err) in enumerate(zip(feat_ref, feat_tgt, feat_err)):
        print("Error is", err[0])
        a, b = ref.ravel()
        c, d = tgt.ravel()
        cv2.line(regImg, (a, b), (c, d), [255, 255, 255], 2)
        if err < 0.8 and feat_found[i] == 1:
            cv2.circle(regImg, (a, b), 5, [255, 0, 0], -1)
            cv2.circle(tgtImg, (c, d), 5, [255, 0, 0], -1)

    # sticker points on feature points.
    p0 = [211, 432]
    p1 = [211, 512]
    p2 = [275, 512]
    p3 = [275, 432]

    # define reference points
    ref_corners = np.array([p0, p1, p2, p3], np.float32)

    # check if it analyzes global motion or local motion
    fctrX = 0.25 * (p0[0] + p1[0] + p2[0] + p3[0])
    fctrY = 0.25 * (p0[1] + p1[1] + p2[1] + p3[1])
    fradius = 40.0  # the radius of the sticker

    vecdistnregion = []
    for ref in feat_ref:
        fdist = math.sqrt((ref[0][0] - fctrX) * (ref[0][0] - fctrX) + (ref[0][1] - fctrY) * (ref[0][1] - fctrY))
        nratio = int(fdist / fradius + 0.499999)
        vecdistnregion.append((fdist, nratio))

    # Check to see if the sticker has feature points within its radius
    numfeatPtsonSticker = 0
    for vec in vecdistnregion:
        first, second = vec
        if (second == 1):
            numfeatPtsonSticker += 1

    matchedptsref = []
    matchedptstgt = []
    if (numfeatPtsonSticker > 6):
        # Estimate Rigid Transform
        for i, (ref, tgt, err) in enumerate(zip(feat_ref, feat_tgt, feat_err)):
            vec = vecdistnregion[i]
            if err[0] < 0.5 and feat_found[i][0] == 1 and vec[1] == 1:
                matchedptsref.append(ref[0])
                matchedptstgt.append(tgt[0])
    else:
        # Estimate Rigid Transform
        for i, (ref, tgt, err) in enumerate(zip(feat_ref, feat_tgt, feat_err)):
            if err[0] < 0.5 and feat_found[i][0] == 1:
                matchedptsref.append(ref[0])
                matchedptstgt.append(tgt[0])

    H, mask = cv2.findHomography(np.array(matchedptsref), np.array(matchedptstgt), cv2.RANSAC)
    tgt_corner = cv2.perspectiveTransform(np.array([ref_corners]), H)
    print(tgt_corner)
    cv2.line(regImg, (p0[0], p0[1]), (p1[0], p1[1]), (255, 0, 0), 2)
    cv2.line(regImg, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 2)
    cv2.line(regImg, (p2[0], p2[1]), (p3[0], p3[1]), (255, 0, 0), 2)
    cv2.line(regImg, (p3[0], p3[1]), (p0[0], p0[1]), (255, 0, 0), 2)

    tp0 = tgt_corner[0][0]
    tp1 = tgt_corner[0][1]
    tp2 = tgt_corner[0][2]
    tp3 = tgt_corner[0][3];

    cv2.line(tgtImg, (tp0[0], tp0[1]), (tp1[0], tp1[1]), (255, 0, 0), 2)
    cv2.line(tgtImg, (tp1[0], tp1[1]), (tp2[0], tp2[1]), (255, 0, 0), 2)
    cv2.line(tgtImg, (tp2[0], tp2[1]), (tp3[0], tp3[1]), (255, 0, 0), 2)
    cv2.line(tgtImg, (tp3[0], tp3[1]), (tp0[0], tp0[1]), (255, 0, 0), 2)

    cv2.imshow("reference", regImg);
    cv2.imshow("target", tgtImg);

    cv2.waitKey(0);


testOpticalFlow()
