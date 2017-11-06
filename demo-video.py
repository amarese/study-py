import numpy as np
import cv2
import math


def testOpticalFlow():
    cap = cv2.VideoCapture('images/model.mp4')

    # Take first frame and find corners in it
    ret, regImg = cap.read()
    regImggray = cv2.cvtColor(regImg, cv2.COLOR_BGR2GRAY)

    feature_params = dict(maxCorners=50,
                          qualityLevel=0.01,
                          minDistance=20,
                          blockSize=3,
                          useHarrisDetector=False,
                          k=0.04)

    # get good features for
    feat_ref = cv2.goodFeaturesToTrack(regImggray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(regImg)

    # sticker points on feature points.
    p0 = [411, 432]
    p1 = [411, 512]
    p2 = [475, 512]
    p3 = [475, 432]

    # define reference points
    ref_corners = np.array([p0, p1, p2, p3], np.float32)
    ref_corner = np.array([ref_corners])

    while (1):
        ret,tgtImg = cap.read()
        tgtImggray = cv2.cvtColor(tgtImg, cv2.COLOR_BGR2GRAY)
        feat_tgt = cv2.goodFeaturesToTrack(tgtImggray, mask=None, **feature_params)

        lk_params = dict(winSize=(15, 15),
                         maxLevel=5,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.3),
                         flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                         minEigThreshold=1e-5)
        feat_tgt, feat_found, feat_err = cv2.calcOpticalFlowPyrLK(regImggray, tgtImggray, feat_ref, feat_tgt, **lk_params)

        # draw the tracks
        # for i, (ref, tgt, err) in enumerate(zip(feat_ref, feat_tgt, feat_err)):
        #     print("Error is", err[0])
        #     a, b = ref.ravel()
        #     c, d = tgt.ravel()
        #     cv2.line(mask, (a, b), (c, d), [255, 255, 255], 2)
        #     if err < 0.8 and feat_found[i] == 1:
        #         cv2.circle(mask, (a, b), 5, [255, 0, 0], -1)
        #         cv2.circle(mask, (c, d), 5, [255, 0, 0], -1)

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

        H, _ = cv2.findHomography(np.array(matchedptsref), np.array(matchedptstgt), cv2.RANSAC)
        tgt_corner = cv2.perspectiveTransform(ref_corner, H)

        rp0 = ref_corner[0][0]
        rp1 = ref_corner[0][1]
        rp2 = ref_corner[0][2]
        rp3 = ref_corner[0][3]

        cv2.line(mask, (rp0[0], rp0[1]), (rp1[0], rp1[1]), (255, 0, 0), 2)
        cv2.line(mask, (rp1[0], rp1[1]), (rp2[0], rp2[1]), (255, 0, 0), 2)
        cv2.line(mask, (rp2[0], rp2[1]), (rp3[0], rp3[1]), (255, 0, 0), 2)
        cv2.line(mask, (rp3[0], rp3[1]), (rp0[0], rp0[1]), (255, 0, 0), 2)

        tp0 = tgt_corner[0][0]
        tp1 = tgt_corner[0][1]
        tp2 = tgt_corner[0][2]
        tp3 = tgt_corner[0][3]

        cv2.line(mask, (tp0[0], tp0[1]), (tp1[0], tp1[1]), (255, 0, 0), 2)
        cv2.line(mask, (tp1[0], tp1[1]), (tp2[0], tp2[1]), (255, 0, 0), 2)
        cv2.line(mask, (tp2[0], tp2[1]), (tp3[0], tp3[1]), (255, 0, 0), 2)
        cv2.line(mask, (tp3[0], tp3[1]), (tp0[0], tp0[1]), (255, 0, 0), 2)

        img = cv2.add(tgtImg, mask)
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        regImggray = tgtImggray.copy()
        feat_ref = feat_tgt.copy()
        ref_corner = tgt_corner.copy()
        mask = np.zeros_like(regImg)

testOpticalFlow()
