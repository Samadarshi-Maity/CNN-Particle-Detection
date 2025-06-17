# this file contains the methodology to compute the evaluation metrics for the detection. 

# import the necessary modules 
import numpy as np
from sklearn.neighbors import NearestNeighbors

class metrics:
    '''
    This class is used to ccompute the evaluation metrics (confusion matrix). 
    Following are the definitions: 
         True Positive- TP: particles that are correctly detected
         False Positive-FP: Particle is detected when it doesn't exist
         False Negative- FN: Particle is not detected when it exists.
         True Negative doe not make any general sense in this case hence omitted from the computation.
         
    The class uses KNN  to seach for the correct pair of true and detected centers within a radius of 3 pixels tocreate the aforementioned classes
    '''
    def __init__(self, x_coord, y_coord, x_gt, y_gt):
        '''
        Constructor to create the object and inputs the predicted coordinates and the ground truth

        Params: 
            x_coord :  x coordinates of the particles detected 
            y_coord :  y coordinates of the particles detected 
            x_gt    :  x coordinates of ground truth 
            y_gt    :  y coordinates of ground truth
        '''
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.x_gt    = x_gt
        self.y_gt    = y_gt
        
    def KNN_for_tagging(self):

        '''
        Employs the KNN aprroach to make the pairs.    
        ''' 

        # fit a KNN with K =1
        coord_gt = np.array([self.x_gt, self.y_gt]).transpose()
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(coord_gt)

        coord_xy = np.array([self.x_coord, self.y_coord]).transpose()
        # Find nearest neighbor in predictions for each point in ground truth
        self.distances, _ = knn.kneighbors(coord_xy)


    def metrics(self):
        '''
        Computes the confusion matrix using the aforementioned defintions
        '''
        # computes the True Positives
        count = len(np.where((self.distances.flatten()<2) & (self.distances.flatten()>0))[0])

        # false Positives
        FP  = len(self.x_coord) - count

        # False negatives
        FN = len(self.x_gt) - count
         
        # compute the precision    
        precision = count/(count + FP)

        # compute the recall
        recall = count/(count + FN)
        
        # return the confusion matrix. ...
        return {"precision": np.round(precision, 3),
                'recall'   : np.round(recall, 3),
                'F1'       : np.round(2 * precision * recall / (precision + recall), 3),
                'accuracy' : np.round(count / (count + FP + FN), 3)}   # concept of true negative is irrelevant here.
                