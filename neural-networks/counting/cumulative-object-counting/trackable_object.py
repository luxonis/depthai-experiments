# from https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]

        self.counted = False
