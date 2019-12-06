import numpy as np
from focusSimulation import blurImage
import math
import cv2
from matplotlib import pyplot as plt

MAX_FOCUSER_POSITION = 255


class FocuserPosition:
    """
    Basic class to capture focuser position and its respective focus score
    """

    def __init__(self, position, score):
        self.p = position
        self.s = score


def getMagnitudeSpectrum(blurredImage):
    """
    Calculate the FFT using openCV. Return the top left quadrant.

    :param blurredImage: Image to transform
    :return: Top left corner of FFT
    """
    # First convert to gray before transforming
    ftransform = np.fft.fft2(cv2.cvtColor(blurredImage, cv2.COLOR_RGB2GRAY))
    # We only need one quadrant of the FFT
    ftransform = np.vsplit(np.hsplit(ftransform, 2)[0], 2)[0]
    return np.log(np.abs(ftransform)) / 20


def getFocusScore(magnitudeSpectrum):
    """
    The focus value is calculated by taking the pixel position in the magnitude, shifting extra weight to the high
    frequency components by taking the sum of the natural log of its position, and multiplying that by the pixel value

    :param magnitudeSpectrum: spectrum returned by getMagnitudeSpectrum()
    :return: Abstract value for how focused image is
    """
    height, width = magnitudeSpectrum.shape
    accumulator = 0.0
    for row in range(0, height):
        for col in range(0, width):
            logRow = 1 if row == 0 else math.log(row)
            logCol = 1 if col == 0 else math.log(col)
            accumulator += magnitudeSpectrum.item(row, col) * (logRow + logCol)

    return accumulator / (height * width)


def getSubArea(img, hCenter, wCenter, size):
    radius = math.floor(size / 2)
    hAreaSplit = np.hsplit(img, [wCenter - radius, wCenter + size - radius])[1]
    return np.vsplit(hAreaSplit, [hCenter - radius, hCenter + size - radius])[1]


def getCenter50(img):
    height, width = img.shape[:2]
    return getSubArea(img, round(height / 2), round(width / 2), 50)


class AutofocusAlgorithm:
    def __init__(self, sceneProperties, focuserProperties, img):
        self.sProp = sceneProperties
        self.fProp = focuserProperties
        self.originalImage = img
        self.fPosMaxDelta = math.floor(self.fProp.maxBarrelSpeed * 255.0 / self.fProp.barrelLength)
        self.fPos = 0  # Current capture position. Move before current capture
        self.curCaptureIndex = 0  # Increment after current capture
        self.pList = []

    def displayImage(self, img, centerChunk, magnitudeSpectrum, blurRadius, finalImage=False):
        if self.curCaptureIndex == self.fProp.maxNumCaptures - 1:
            finalImage = True

        plt.figure(figsize=(15, 8))
        plt.subplot(131)
        plt.imshow(img)
        plt.title('Blur Radius = ' + str(blurRadius))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(132)
        plt.imshow(centerChunk)
        if finalImage:
            appendThis = '(FINAL IMAGE)'
        else:
            appendThis = ''
        plt.title('Capture = ' + str(self.curCaptureIndex) + ', focuser position = ' + str(self.fPos) + appendThis)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(133)
        plt.imshow(magnitudeSpectrum, cmap='gray')
        plt.title('Magnitude Spectrum = ' + str(getFocusScore(magnitudeSpectrum)))
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def insertIntoList(self, pos, score):
        """
        Keeps the pList sorted

        :param pos: Focuser position
        :param score: Focuser score
        :return:
        """
        for i in range(0, len(self.pList)):
            if self.pList[i].p > pos:
                self.pList.insert(i, FocuserPosition(pos, score))
                self.curCaptureIndex += 1
                return
            elif self.pList[i].p == pos:
                print('same!')
                # The program is deterministic, so we will skip this for now
                if self.pList[i].s != score:
                    print('WARNING: pos(' + str(pos) + ') may have different values: ' +
                          str(self.pList[i].s) + ', ' + str(score))
                self.curCaptureIndex += 1
                return

        self.pList.append(FocuserPosition(pos, score))
        self.curCaptureIndex += 1

    def getMaxFocusScore(self):
        maxScore = 0
        maxScoreIndex = 0
        for i in range(0, len(self.pList)):
            if self.pList[i].s > maxScore:
                maxScore = self.pList[i].s
                maxScoreIndex = i
        return maxScore, maxScoreIndex

    def moveFocuserToPosition(self, pos):
        if self.fPos < pos:
            self.fPos = min(pos, self.fPos + self.fPosMaxDelta)
        else:
            self.fPos = max(pos, self.fPos - self.fPosMaxDelta)
        return self.fPos

    def focusImage(self, display=True):
        # Number of captures needed to sweep length of barrel
        capturesToSweepBarrel = math.ceil(MAX_FOCUSER_POSITION * 1.0 / self.fPosMaxDelta)

        while self.curCaptureIndex < self.fProp.maxNumCaptures - 1:
            # If this is not the first capture, increment fPos
            if len(self.pList) != 0:
                self.fPos = min(self.fPos + self.fPosMaxDelta, MAX_FOCUSER_POSITION)

            # Get blurred image
            blurredImage, blurRadius = blurImage(self.sProp, self.fProp, self.fPos, self.originalImage)
            centerChunk = getCenter50(blurredImage)
            magnitudeSpectrum = getMagnitudeSpectrum(centerChunk)
            fScore = getFocusScore(magnitudeSpectrum)
            if display:
                self.displayImage(blurredImage, centerChunk, magnitudeSpectrum, blurRadius)

            # Append to pList
            self.insertIntoList(self.fPos, fScore)

            # If we are not given enough captures to sweep entire barrel, we make the following assumption:
            #     If the focus score increases at any point, we have passed convergence
            print('Captures to sweep barrel', capturesToSweepBarrel)
            print('Maxnumcaptures', self.fProp.maxNumCaptures)
            if math.ceil(capturesToSweepBarrel * 4 / 3 + 1) >= self.fProp.maxNumCaptures:
                # If focus score at last position is better than what it is now.
                # We add a threshold of 0.1 to get rid of noise
                print('!!!')
                if len(self.pList) > 1 and self.pList[-2].s > fScore + 0.1:
                    # If position is past the 2/3 point, we can afford to do a full sweep. Otherwise we need to call
                    # it quits
                    if self.pList[-2].p < math.floor(MAX_FOCUSER_POSITION * 2 / 3):
                        break

            # We performed a full sweep at this point
            if self.fPos == MAX_FOCUSER_POSITION:
                break

        # We select the best focused image and its neighbors. Convergence should be somewhere within
        maxScore, maxScoreIndex = self.getMaxFocusScore()

        # We must move back to position of maxScoreIndex + 1. We can keep track on the way back, and sanity check.
        # But this should never be the highest value.
        upperThreshold = min(maxScoreIndex + 1, len(self.pList) - 1)
        innerSweepStart = self.pList[upperThreshold].p
        print('innerSweepStart = ', innerSweepStart)

        # We want to save one more move at the very end to adjust to highest known focus score location
        while innerSweepStart < self.fPos and self.curCaptureIndex < self.fProp.maxNumCaptures - 1:
            self.fPos -= self.fPosMaxDelta

            # Get blurred image
            blurredImage, blurRadius = blurImage(self.sProp, self.fProp, self.fPos, self.originalImage)
            centerChunk = getCenter50(blurredImage)
            magnitudeSpectrum = getMagnitudeSpectrum(centerChunk)
            fScore = getFocusScore(magnitudeSpectrum)
            if display:
                self.displayImage(blurredImage, centerChunk, magnitudeSpectrum, blurRadius)

            self.insertIntoList(self.fPos, fScore)

        # Now that we are within one fPosMaxDelta to the current highest score. The strategy now is as follows:
        #     For each iteration, we take one point below and one point above current focus score max.
        #     We move the focuser to 1/3 between the two points and sample again (closer to the point closer to the
        #     current focuser position)

        # We want to save one more move at the very end to adjust to highest known focus score location
        while self.curCaptureIndex < self.fProp.maxNumCaptures - 1:
            # First, find the best focus score
            maxScore, maxScoreIndex = self.getMaxFocusScore()

            print('best score location = ', self.pList[maxScoreIndex].p)

            thresholdLowIndex = max(0, maxScoreIndex - 1)
            thresholdHighIndex = min(len(self.pList) - 1, maxScoreIndex + 1)

            closerIndex = thresholdLowIndex if abs(self.fPos - self.pList[thresholdLowIndex].p) < \
                                               abs(self.fPos - self.pList[thresholdHighIndex].p) else thresholdHighIndex
            furtherIndex = thresholdLowIndex if closerIndex == thresholdHighIndex else thresholdHighIndex
            nextPosition = round((self.pList[closerIndex].p * 2 + self.pList[furtherIndex].p) / 3)

            # Used to determine convergence at the end of while statement
            curPosition = self.fPos

            self.moveFocuserToPosition(nextPosition)

            # We can consider the algorithm converged if the bottom criteria is met (actual capture is done after
            # this for loop
            if abs(maxScore - fScore) < 0.1 and abs(curPosition - self.fPos) < 5:
                break

            # Get blurred image
            blurredImage, blurRadius = blurImage(self.sProp, self.fProp, self.fPos, self.originalImage)
            centerChunk = getCenter50(blurredImage)
            magnitudeSpectrum = getMagnitudeSpectrum(centerChunk)
            fScore = getFocusScore(magnitudeSpectrum)
            if display:
                self.displayImage(blurredImage, centerChunk, magnitudeSpectrum, blurRadius)

            self.insertIntoList(self.fPos, fScore)


        # Final adjustment towards the best score
        maxScore, maxScoreIndex = self.getMaxFocusScore()
        self.moveFocuserToPosition(self.pList[maxScoreIndex].p)

        # Get blurred image
        blurredImage, blurRadius = blurImage(self.sProp, self.fProp, self.fPos, self.originalImage)
        centerChunk = getCenter50(blurredImage)
        magnitudeSpectrum = getMagnitudeSpectrum(centerChunk)
        fScore = getFocusScore(magnitudeSpectrum)
        if display:
            self.displayImage(blurredImage, centerChunk, magnitudeSpectrum, blurRadius, True)

        self.insertIntoList(self.fPos, fScore)

