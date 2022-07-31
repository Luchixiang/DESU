import os
import cv2

img_path = '/Users/Luchixiang/Downloads/vis/full_400fake/test_15_noiselevel1_noisy_2.png'
gt_path = img_path.replace('noisy', 'gt')
transfer_path = img_path.replace('noisy', 'denoised')
similar_path = '/Users/Luchixiang/Downloads/vis/similar_400fake/test_15_noiselevel1_denoised_2.png'
dissimilar_path = '/Users/Luchixiang/Downloads/vis/dissimilar_100fake/test_15_noiselevel1_denoised_2.png'
scratch_path = '/Users/Luchixiang/Downloads/vis/scratch_100fake/test_15_noiselevel1_denoised_2.png'


target_path = '/Users/Luchixiang/Downloads/zoomed'
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)

        print(x, y)



        extracted_img = img[y-25: y + 25, x-25: x + 25, :]
        extracted_img = cv2.resize(extracted_img, (128, 128))
        cv2.rectangle(img, (x - 25, y - 25), (x + 25, y + 25), (0, 0, 255))

        cv2.imwrite(os.path.join(target_path, os.path.basename(img_path)), img)
        cv2.imwrite(os.path.join(target_path, os.path.basename(img_path).replace('.png', 'noisy_zoomed.png')), extracted_img)

        cv2.imshow("image", img)
        cv2.imshow('zoomed', extracted_img)
        # cv2.imshow('gt', gt_img)
        gt_extracted = gt_img[y-25: y + 25, x-25: x + 25, :]
        gt_extracted = cv2.resize(gt_extracted, (128, 128))
        transfer_extracted = transfer_img[y-25: y + 25, x-25: x + 25, :]
        transfer_extracted = cv2.resize(transfer_extracted, (128, 128))
        scratch_extracted = scratch_img[y - 25: y + 25, x - 25: x + 25, :]
        scratch_extracted = cv2.resize(scratch_extracted, (128, 128))
        dissimlar_extracted = dissimilar_img[y - 25: y + 25, x - 25: x + 25, :]
        dissimlar_extracted = cv2.resize(dissimlar_extracted, (128, 128))
        similar_extracted = similar_img[y - 25: y + 25, x - 25: x + 25, :]
        similar_extracted = cv2.resize(similar_extracted, (128, 128))
        cv2.imwrite(os.path.join(target_path, os.path.basename(img_path).replace('.png', 'gt_zoomed.png')), gt_extracted)
        cv2.imwrite(os.path.join(target_path, os.path.basename(img_path).replace('.png', 'transfer_zoomed.png')), transfer_extracted)
        cv2.imwrite(os.path.join(target_path, os.path.basename(img_path).replace('.png', 'scratch_zoomed.png')), scratch_extracted)
        cv2.imwrite(os.path.join(target_path, os.path.basename(img_path).replace('.png', 'similar_zoomed.png')), similar_extracted)
        cv2.imwrite(os.path.join(target_path, os.path.basename(img_path).replace('.png', 'dissimilar_zoomed.png')), dissimlar_extracted)
        cv2.imshow('gt_extracted', gt_extracted)
        cv2.imshow('transfer_extracted', transfer_extracted)

img = cv2.imread(img_path)
gt_img = cv2.imread(gt_path)
transfer_img = cv2.imread(transfer_path)
similar_img = cv2.imread(similar_path)
dissimilar_img = cv2.imread(dissimilar_path)
scratch_img = cv2.imread(scratch_path)
print(img.shape)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", img)
cv2.imshow("gt", gt_img)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
key = cv2.waitKey()
if key == ord('q'):
    cv2.destroyAllWindows()