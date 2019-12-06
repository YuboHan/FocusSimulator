# from PIL import Image
# from PIL import ImageFilter
#
# im = Image.open('../images/forest.jpg')
# out = im.filter(ImageFilter.GaussianBlur(5))
#
# out.show()

import cv2
import math


def calculateRho(sceneProperties, focuserProperties, img):
    """
    :param sceneProperties: SceneProperties object
    :param focuserProperties: FocuserProperties object
    :param img: numpy image shape (get using image.shape)
    :return: Rho of blur circle
    """
    realSceneHeight = math.tan(math.radians(focuserProperties.viewAngle / 2.0)) * sceneProperties.distance * 2
    equivalentSensorHeight = realSceneHeight * focuserProperties.focalLength / sceneProperties.distance
    return max(img.shape[0], img.shape[1]) / equivalentSensorHeight


def getBlurRadius(sceneProperties, focuserProperties, focuserPosition):
    f = focuserProperties.focalLength
    u = sceneProperties.distance
    s = f + (focuserPosition / 255.0) * focuserProperties.barrelLength
    A = focuserProperties.aperture

    r = s * A * ((1.0 / f) - (1.0 / u) - (1.0 / s)) / 2

    return r


def getPixelBlurRadius(sceneProperties, focuserProperties, focuserPosition, img):
    r = getBlurRadius(sceneProperties, focuserProperties, focuserPosition)
    p = calculateRho(sceneProperties, focuserProperties, img)

    return r * p


def blurImage(sceneProperties, focuserProperties, focuserPosition, img):
    blurRadius = round(getPixelBlurRadius(sceneProperties, focuserProperties, focuserPosition, img))
    blurDiameter = blurRadius * 2 + 1

    return cv2.GaussianBlur(img, (blurDiameter, blurDiameter), cv2.BORDER_DEFAULT), blurRadius

# Defaults:
# focalLength = .05
# fNumber = 1.8
# Aperture = 0.028

# if __name__ == "__main__":
#     defaultScene = SceneProperties(3)
#     defaultFocuser = FocuserProperties(0.028, 0.05, 100, 0.01, 10, 0.0025)
#
#     forestImage = cv2.imread('../images/stanford.jpg')
#     forestGray = cv2.cvtColor(forestImage, cv2.COLOR_BGR2GRAY)
#
#     for i in range(0, 16):
#         fp = i * 16
#         blurRadius = getPixelBlurRadius(defaultScene, defaultFocuser, fp, forestGray)
#         blurDiameter = round(abs(blurRadius)) * 2 + 1
#         blurredImage = cv2.GaussianBlur(forestImage, (blurDiameter, blurDiameter), cv2.BORDER_DEFAULT)
#
#         print('fp = ' + str(fp))
#         print('blur diameter = ' + str(blurDiameter))
#
#         plt.figure(figsize=(15,8))
#         plt.subplot(121), plt.imshow(blurredImage)
#         plt.title('Input image'), plt.xticks([]), plt.yticks([])
#
#         ft = np.fft.fft2(cv2.cvtColor(blurredImage, cv2.COLOR_BGR2GRAY)) # FFT transform
#         ft = np.fft.fftshift(ft)
#         magnitude_spectrum = np.log(np.abs(ft)) / 20
#
#         plt.subplot(122), plt.imshow(magnitude_spectrum, cmap = 'gray')
#         plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#         plt.show()
#
#         n = 0
#         for num in magnitude_spectrum:
#             for nu in num:
#                 n += nu
#
#         print(n)

# print('i = ' + str(fp) + ', ' + str(getPixelBlurRadius(defaultScene, defaultFocuser, fp, forestImage)))


# forestImage = cv2.imread('../images/forest.jpg')
# stanfordImage = cv2.imread('../images/stanford.jpg')
#
# forestGray = cv2.cvtColor(forestImage, cv2.COLOR_BGR2GRAY)
# forestGray = cv2.GaussianBlur(forestGray, (31,31), cv2.BORDER_DEFAULT)
#
# f = np.fft.fft2(forestGray) # FFT transform
# f = np.fft.fftshift(f)
#
# #magnitude_spectrum = 20 * np.log(np.abs(f))
# magnitude_spectrum = np.log(np.abs(f)) / 20
#
# print(magnitude_spectrum)
#
# plt.subplot(121), plt.imshow(forestGray, cmap = 'gray')
# plt.title('Input image'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(122), plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

# cv2.namedWindow('image')
# cv2.imshow('image', img)
# cv2.waitKey(100)

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#
# while (True):
#     cv2.imshow('image', forestImage)
#     cv2.waitKey(500)
#     cv2.imshow('image', stanfordImage)
#     cv2.waitKey(500)
