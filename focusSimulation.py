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
    if focuserPosition > 255:
        raise Exception('Invalid focuser position ' + str(focuserPosition) + '. Must be [0, 255].')

    blurRadius = abs(round(getPixelBlurRadius(sceneProperties, focuserProperties, focuserPosition, img)))
    blurDiameter = blurRadius * 2 + 1

    return cv2.GaussianBlur(img, (blurDiameter, blurDiameter), cv2.BORDER_DEFAULT), blurRadius
