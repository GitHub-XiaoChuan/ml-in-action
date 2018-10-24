import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import walk
from os.path import join
import sys

# 书中的代码

# def create_descriptors(folder):
#     files = []
#     for (dirpath, dirnames, filenames) in walk(folder):
#         files.extend(filenames)
#     for f in files:
#         save_descriptor(folder, f, cv2.xfeatures2d.SIFT_create())
#
#
# def save_descriptor(folder, image_path, feature_detector):
#     img = cv2.imread(join(folder, image_path), 0)
#     keypoints, descriptors = feature_detector.detectAndCompute(img, None)
#     descriptor_file = image_path.replace("jpg", "npy")
#     np.save(join(folder, descriptor_file), descriptors)


# dir = sys.argv[1]
# create_descriptors(dir)

# ------------------------------------------

# folder = sys.argv[1]
# query = cv2.imread('product_adidas.jpg', 0)
query = cv2.imread('product_adidas.jpg', 0)

logs = ['logo_adidas.jpg', 'logo_nike.jpg']
files = []
images = []
descriptors = {}

orb = cv2.ORB_create()
for image_path in logs:
    img = cv2.imread(image_path, 0)
    kp, des = orb.detectAndCompute(img, None)
    descriptors[image_path] = (des, kp, img)

# for (dirpath, dirname, filenames) in walk(folder):
#     files.extend(filenames)
#     for f in files:
#         if f.endswith("npy") and f!="tatto_seed.npy":
#             descriptors.append(f)
#     print(descriptors)

# sift = cv2.xfeatures2d.SIFT_create()
# query_kp, query_ds = sift.detectAndCompute(query, None)

query_kp, query_ds = orb.detectAndCompute(query, None)

# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
FLANN_INDEX_LSH = 6

index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,  # 20
                    multi_probe_level=1)  # 2
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

MIN_MATCH_COUNT = 10

potential_culprits = {}

print(">> Initiating picture scan ...")
for name, (des, kp, img) in descriptors.items():
    print("------- analyzing %s for matches ------" % name)
    matches = flann.knnMatch(query_ds, des, k=2)
    good = []
    for x in matches:
        if len(x) == 2 and x[0].distance < 0.7 * x[1].distance:
            good.append(x[0])
    if len(good) >= MIN_MATCH_COUNT:
        print("%s is a match! (%d)" % (name, len(good)))
    else:
        print("%s is not a match" % name)
    potential_culprits[name] = len(good)

max_matches = None
potential_suspect = None

for culprit, matches in potential_culprits.items():
    if max_matches == None or matches > max_matches:
        max_matches = matches
        potential_suspect = culprit

print("potential suspect is %s" % potential_suspect.upper())

img2 = descriptors[potential_suspect][2]
kp2 = descriptors[potential_suspect][1]

if len(good) >= MIN_MATCH_COUNT:
    # 寻找关键点
    src_pts = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([query_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 单应性
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = query.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

drawParams = dict(matchColor=(0, 255, 0),
                  singlePointColor=(255, 0, 0),
                  matchesMask=matchesMask,
                  flags=2)

img3 = cv2.drawMatches(query, query_kp, img2, kp2, good, None, **drawParams)
plt.imshow(img3, 'gray')
plt.show()