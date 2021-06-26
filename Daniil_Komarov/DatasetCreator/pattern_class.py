class RoadSignDetectorPattern:
  def __init__(self, draw_kps):
    self.test_img = []
    self.query_img = None
    self.draw_kps = draw_kps
    self.latestFileId = 0

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

  def addTrainImage(self, train_img, name):
    pass

  def createVideo(self, images, framerate, path):
    pass