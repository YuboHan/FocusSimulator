class SceneProperties:
    def __init__(self, distance):
        if distance < .1:
            raise Exception('Minimum scene distance must be more than 0.1 meters away')
        self.distance = distance


class FocuserProperties:
    def __init__(self, aperture, focalLength, viewAngle, barrelLength, maxNumCaptures, maxBarrelSpeed):
        self.aperture = aperture
        self.focalLength = focalLength
        self.viewAngle = viewAngle
        self.barrelLength = barrelLength
        self.maxNumCaptures = maxNumCaptures
        self.maxBarrelSpeed = maxBarrelSpeed