class RoadSignDetectorPattern:
  def __init__(self, test_img, query_img):
    self.test_img = test_img
    self.query_img = query_img

  def detect(self):
    pass

  def compare(self, test_kps, test_des, query_kps, query_des):
    pass
  
  def drawKps(self, query_kps, good_points):
    pass

  def calculateSignCenters(self, query_kps, good_points):
    pass

  def run(self):
    pass

  def changeQueryImage(self, query_img):
    pass

  def createVideo(self, images):
    pass