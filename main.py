import focusSimulation as fs
from common import SceneProperties
from common import FocuserProperties
import numpy as np
from matplotlib import pyplot as plt
import cv2

if __name__ == "__main__":
    defaultScene = SceneProperties(3)

    # Defaults:
    # focalLength = .05
    # fNumber = 1.8
    # Aperture = 0.028
    defaultFocuser = FocuserProperties(0.028, 0.05, 100, 0.01, 10, 0.0025)

    forestImage = cv2.imread('images/stanford.jpg')
    if forestImage is None:
        raise Exception('Cannot find image')

    for i in range(0, 16):
        fp = i * 16

        blurredImage, blurRadius = fs.blurImage(defaultScene, defaultFocuser, fp, forestImage)

        # Get Fourier transform
        ftransform = np.fft.fft2(cv2.cvtColor(blurredImage, cv2.COLOR_RGB2GRAY))
        ftransform = np.fft.fftshift(ftransform)
        magnitude_spectrum = np.log(np.abs(ftransform)) / 20

        plt.figure(figsize=(15,8))
        plt.subplot(121)
        plt.imshow(blurredImage)
        plt.title('Blur Radius = ' + str(blurRadius))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(122)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum')
        plt.xticks([])
        plt.yticks([])
        plt.show()


    # for i in range(0, 16):
    #     fp = i * 16
    #     blurRadius = focusSimulation.getPixelBlurRadius(defaultScene, defaultFocuser, fp, forestGray)
    #     blurDiameter = round(abs(blurRadius)) * 2 + 1
    #     blurredImage = cv2.GaussianBlur(forestImage, (blurDiameter, blurDiameter), cv2.BORDER_DEFAULT)
    #
    #     print('fp = ' + str(fp))
    #     print('blur diameter = ' + str(blurDiameter))
    #
    #     plt.figure(figsize=(15,8))
    #     plt.subplot(121), plt.imshow(blurredImage)
    #     plt.title('Input image'), plt.xticks([]), plt.yticks([])
    #
    #     ft = np.fft.fft2(cv2.cvtColor(blurredImage, cv2.COLOR_BGR2GRAY)) # FFT transform
    #     ft = np.fft.fftshift(ft)
    #     magnitude_spectrum = np.log(np.abs(ft)) / 20
    #
    #     plt.subplot(122), plt.imshow(magnitude_spectrum, cmap = 'gray')
    #     plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    #     plt.show()
    #
    #     n = 0
    #     for num in magnitude_spectrum:
    #         for nu in num:
    #             n += nu
    #
    #     print(n)