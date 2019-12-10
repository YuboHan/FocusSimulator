from common import SceneProperties
from common import FocuserProperties
from autofocusAlgorithm import AutofocusAlgorithm
import cv2

if __name__ == "__main__":
    defaultScene = SceneProperties(1)

    # Defaults:
    # focalLength = .05
    # fNumber = 1.8
    # Aperture = 0.028
    # defaultFocuser = FocuserProperties(0.028, 0.05, 100, 0.01, 10, 0.0025)

    defaultFocuser = FocuserProperties(0.028, 0.05, 100, 0.01, 20, 0.0015)

    forestImage = cv2.imread('images/stanford.jpg')
    if forestImage is None:
        raise Exception('Cannot find image')

    forestImage = cv2.cvtColor(forestImage, cv2.COLOR_BGR2RGB)

    focuser = AutofocusAlgorithm(defaultScene, defaultFocuser, forestImage)
    focuser.focusImage()
