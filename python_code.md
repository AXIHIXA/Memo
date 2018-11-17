# python code

## opencv-python + matplotlib

    import cv2
    import json
    import matplotlib
    import matplotlib.pylab as plt
    import numpy as np
    import os
    import sys


    def compute_ious(bboxes1, bboxes2):
        """
        Compute ious of two groups of bounding boxes
        :param bboxes1: np.ndarray - size (N1, (x1, y1, x2, y2))
        :param bboxes2: np.ndarray - size (N2, (x1, y1, x2, y2))
        :return: np.ndarray - size (N1, N2)
        """
        ious = []
        for bbox1 in bboxes1:
            bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            # in case there are some zeros.
            if bbox1_area == 0:
                ious.append(np.zeros((len(bboxes2), ), dtype=np.float32))
                continue
            # Calculate intersection areas
            x1 = np.maximum(bbox1[0], bboxes2[:, 0])
            y1 = np.maximum(bbox1[1], bboxes2[:, 1])
            x2 = np.minimum(bbox1[2], bboxes2[:, 2])
            y2 = np.minimum(bbox1[3], bboxes2[:, 3])
            intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
            bboxes2_area = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
            union = bbox1_area + bboxes2_area[:] - intersection[:]
            iou = np.float32(intersection) / union
            ious.append(iou)
        return np.array(ious)
        
    img = cv2.imread('SHION.jpg')

    fig = plt.figure(dpi=300)
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis('on')
    plt.title('image')
    plt.show()
    fig.clear()
    
## file system traversal
 
    for root, dirs, files in os.walk('./some_dir/'):
      # `root`: The directory we are traversing in this loop
      # `dirs`: All directories in `root`
      # `files`: All files in `root`

      for filename in files:
        print(os.path.abspath(filename))
