import os
import cv2
from tifffile import imread
import numpy as np
from PIL import Image

path = '/Users/Luchixiang/Downloads/Archive (2)'
coarse_path = '/Users/Luchixiang/Downloads/coarse_img_point/Archive (1)'

# done = set()
# with open('background.txt', 'r') as fp:
#     for line in fp.readlines():
#         done.add(os.path.dirname(line.strip('\n')))
# with open('not_background.txt', 'r') as fp:
#     for line in fp.readlines():
#         done.add(os.path.dirname(line.strip('\n')))
for home, dirs, files in os.walk(path):
    # if '18MAA_31SP_MAP2' not in home:
    #     # 无法对齐
    #     continue
    # if home in done:
    #     continue
    not_background_list = []
    background_list = []
    for file in files:
        if '2021_18MAA/4M2H_14TS1_MAP2' in home:
            background_list.append(os.path.join(home, file))
            continue
        if 'Tile' in file and file.endswith('.tif'):
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            # cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
            try:
                img = Image.open(os.path.join(home, file))
                img = img.resize((512, 512))
            except:
                continue
            try:
                img2 = Image.open(
                    os.path.join(home.replace('Archive (2)', 'coarse_imgs_point/Archive (2)'), file)).convert('L')
            except:
                img2 = Image.open(os.path.join(home.replace('Archive (2)', 'coarse_imgs_point/Archive (2)'),
                                               file.replace('-000_0.tif', '-000_0-000.tif'))).convert('L')
            img2 = np.array(img2)
            Hori = np.vstack((img, img2))
            print(home, file)
            cv2.imshow("image", Hori)
            # cv2.imshow("image2", img2)
            key = cv2.waitKey()
            if key == ord('n'):
                not_background_list.append(os.path.join(home, file))
                cv2.destroyAllWindows()
            elif key == ord('x'):
                background_list.append(os.path.join(home, file))
                cv2.destroyAllWindows()
            key = cv2.waitKey()
    # with open('background.txt', 'a') as fp:
    #     for file in background_list:
    #         fp.write(file + '\n')
    # with open('not_background.txt', 'a') as fp:
    #     for file in not_background_list:
    #         fp.write(file + '\n')
# 4M2H_34SP_MAP2
