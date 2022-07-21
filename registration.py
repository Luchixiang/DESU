import cv2
import numpy as np
import os

subimage = '/Users/Luchixiang/Downloads/Archive (1)/202109_4M2H_2/4M2H_31SP3_MAP1/Tile_005-004-000_0.tif'
image = '/Users/Luchixiang/Downloads/Archive (1)/202109_4M2H_2/4M2H_31SP3_MAP1/screenshot_003.tif'
from tifffile import imread
subimage = cv2.imread(subimage)
subimage = cv2.resize(subimage, (256, 256))
image = cv2.imread(image)
print(image.shape)
start_x = 612
start_y = 310
len_x =  35
len_y = 23

subplot = image[int(start_y + 4 * 0.8 * len_y) - len_y:int(start_y + 4 * 0.8 * len_y) + 2 * len_y,  int(start_x + 3* len_x * 0.8) - len_x:int(start_x + 3* len_x * 0.8) + 2* len_x, :]
subplot = cv2.resize(subplot, (256 * 3, 256 * 3))
result = cv2.matchTemplate(subplot,subimage,5)
min_max = cv2.minMaxLoc(result)
match_loc = min_max[3]
right_bottom = (match_loc[0] + subimage.shape[1], match_loc[1] + subimage.shape[0])
img_disp = subplot.copy()
print(match_loc)
cv2.rectangle(img_disp, match_loc, right_bottom, (0,255,0), 5, 8, 0 )
cv2.normalize( result, result, 0, 255, cv2.NORM_MINMAX, -1 )
cv2.circle(result, match_loc, 10, (255,0,0), 2 )
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2)
ax[0,0].set_title('img_src')
ax[0, 0].imshow(cv2.cvtColor(subplot, cv2.COLOR_BGR2RGB))
ax[0, 1].set_title('img_templ')
ax[0, 1].imshow(cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB))
ax[1,0].set_title('result')
ax[1,0].imshow(result,'gray')
ax[1,1].set_title('img_disp')
ax[1,1].imshow(cv2.cvtColor(img_disp,cv2.COLOR_BGR2RGB))
plt.show()
# print(match_loc)
# cv2.namedWindow('aa')
# key = cv2.waitKey()
# if key == ord('q'):
#     cv2.destroyAllWindows()
# cv2.imwrite('./test')
#print(np.unravel_index(result.argmax(),result.shape))