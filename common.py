class SceneProperties:
    def __init__(self, distance):
        if distance < .1:
            raise Exception('Minimum scene distance must be more than 0.1 meters away')
        self.distance = distance


class FocuserProperties:
    def __init__(self, aperture, focalLength, viewAngle, barrelLength, maxNumCaptures, maxBarrelSpeed):
        if aperture <= 0:
            raise Exception('Aperture must be positive and non-zero')
        if focalLength <= 0:
            raise Exception('Focal length must be positive and non-zero')
        if viewAngle <= 0 or viewAngle > 180:
            raise Exception('View angle must be between [0, 180]')
        if barrelLength <= 0:
            raise Exception('Barrel Length must be positive and non-zero')
        if maxNumCaptures <= 0:
            raise Exception('Max num captures must be positive and non-zero')
        if maxBarrelSpeed <= 0:
            raise Exception('Max barrel speed must be positive and non-zero')
        self.aperture = aperture
        self.focalLength = focalLength
        self.viewAngle = viewAngle
        self.barrelLength = barrelLength
        self.maxNumCaptures = maxNumCaptures
        self.maxBarrelSpeed = maxBarrelSpeed
