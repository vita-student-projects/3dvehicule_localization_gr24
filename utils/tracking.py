import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import ConvexHull
from numpy import *
from filterpy.kalman import KalmanFilter
import torch
import torch.nn as nn
import os
import sys
from typing import Tuple
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualizer import Visualizer

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

  


class KalmanBox3DTracker(object):
    """
    This class represents the internel state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, bbox3D, info=False):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # coord3d - array of detections [x,y,z,theta,l,w,h]
        # X,Y,Z,theta, l, w, h, dX, dY, dZ
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.kf.F = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])

        # state uncertainty, give high uncertainty to
        self.kf.P[7:, 7:] *= 1000.
        # the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.

        # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
        self.kf.Q[7:, 7:] *= 0.01
        self.kf.x[:7] = bbox3D.reshape((7, 1))

        self.time_since_update = 0
        self.id = KalmanBox3DTracker.count
        KalmanBox3DTracker.count += 1
        self.nfr = 5
        self.history = []
        self.prev_ref = bbox3D
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first
        # detection
        self.age = 0
        self.info = info  # other info

    @property
    def obj_state(self):
        return self.kf.x.flatten()

    def _update_history(self, bbox3D):
        self.history = self.history[1:] + [bbox3D - self.prev_ref]

    def _init_history(self, bbox3D):
        self.history = [bbox3D - self.prev_ref] * self.nfr

    def update(self, bbox3D, info=False):
        """
        Updates the state vector with observed bbox.
        """
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        self.time_since_update = 0

        # orientation correction
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi:
            new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi:
            new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        # if the angle of two theta is not acute angle
        if np.pi / 2.0 < abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi:
                self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi:
                self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to
        # < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        # Update the bbox3D
        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2
        self.info = info
        self.prev_ref = self.kf.x.flatten()[:7]

    def predict(self, update_state: bool = True):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x.flatten()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x.flatten()

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history

class LSTM3DTracker(object):
    """
    This class represents the internel state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, device, lstm, bbox3D, info=None):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # coord3d - array of detections [x,y,z,theta,l,w,h]
        # X,Y,Z,theta, l, w, h, dX, dY, dZ

        self.device = device
        self.lstm = lstm
        self.loc_dim = self.lstm.loc_dim
        self.id = LSTM3DTracker.count
        LSTM3DTracker.count += 1
        self.nfr = 5
        self.hits = 1
        self.hit_streak = 0
        self.time_since_update = 0
        self.init_flag = True
        self.age = 0

        self.obj_state = np.hstack([bbox3D.reshape((7, )), np.zeros((3, ))])
        self.history = np.tile(
            np.zeros_like(bbox3D[:self.loc_dim]), (self.nfr, 1))
        self.ref_history = np.tile(bbox3D[:self.loc_dim], (self.nfr + 1, 1))
        self.avg_angle = bbox3D[3]
        self.avg_dim = np.array(bbox3D[4:])
        self.prev_obs = bbox3D.copy()
        self.prev_ref = bbox3D[:self.loc_dim].copy()
        self.info = info
        self.hidden_pred = self.lstm.init_hidden(self.device)
        self.hidden_ref = self.lstm.init_hidden(self.device)

    @staticmethod
    def fix_alpha(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def update_array(origin_array: np.ndarray,
                     input_array: np.ndarray) -> np.ndarray:
        new_array = origin_array.copy()
        new_array[:-1] = origin_array[1:]
        new_array[-1:] = input_array
        return new_array

    def _update_history(self, bbox3D):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.history = self.update_array(
            self.history, self.ref_history[-1] - self.ref_history[-2])
        # align orientation history
        self.history[:, 3] = self.history[-1, 3]
        self.prev_ref[:self.loc_dim] = self.obj_state[:self.loc_dim]
        if self.loc_dim > 3:
            self.avg_angle = self.fix_alpha(self.ref_history[:,
                                                             3]).mean(axis=0)
            self.avg_dim = self.ref_history.mean(axis=0)[4:]
        else:
            self.avg_angle = self.prev_obs[3]
            self.avg_dim = np.array(self.prev_obs[4:])

    def _init_history(self, bbox3D):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.history = np.tile([self.ref_history[-1] - self.ref_history[-2]],
                               (self.nfr, 1))
        self.prev_ref[:self.loc_dim] = self.obj_state[:self.loc_dim]
        if self.loc_dim > 3:
            self.avg_angle = self.fix_alpha(self.ref_history[:,
                                                             3]).mean(axis=0)
            self.avg_dim = self.ref_history.mean(axis=0)[4:]
        else:
            self.avg_angle = self.prev_obs[3]
            self.avg_dim = np.array(self.prev_obs[4:])

    def update(self, bbox3D, info=np.array([[1]])):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        if self.age == 1:
            self.obj_state[:self.loc_dim] = bbox3D[:self.loc_dim].copy()

        if self.loc_dim > 3:
            # orientation correction
            self.obj_state[3] = self.fix_alpha(self.obj_state[3])
            bbox3D[3] = self.fix_alpha(bbox3D[3])

            # if the angle of two theta is not acute angle
            # make the theta still in the range
            curr_yaw = bbox3D[3]
            if np.pi / 2.0 < abs(curr_yaw -
                                 self.obj_state[3]) < np.pi * 3 / 2.0:
                self.obj_state[3] += np.pi
                if self.obj_state[3] > np.pi:
                    self.obj_state[3] -= np.pi * 2
                if self.obj_state[3] < -np.pi:
                    self.obj_state[3] += np.pi * 2

            # now the angle is acute: < 90 or > 270,
            # convert the case of > 270 to < 90
            if abs(curr_yaw - self.obj_state[3]) >= np.pi * 3 / 2.0:
                if curr_yaw > 0:
                    self.obj_state[3] += np.pi * 2
                else:
                    self.obj_state[3] -= np.pi * 2

        with torch.no_grad():
            refined_loc, self.hidden_ref = self.lstm.refine(
                torch.from_numpy(self.obj_state[:self.loc_dim]).view(
                    1, self.loc_dim).float().to(self.device),
                torch.from_numpy(bbox3D[:self.loc_dim]).view(
                    1, self.loc_dim).float().to(self.device),
                torch.from_numpy(self.prev_ref[:self.loc_dim]).view(
                    1, self.loc_dim).float().to(self.device),
                torch.from_numpy(info).view(1, 1).float().to(self.device),
                self.hidden_ref)

        refined_obj = refined_loc.cpu().numpy().flatten()
        if self.loc_dim > 3:
            refined_obj[3] = self.fix_alpha(refined_obj[3])

        self.obj_state[:self.loc_dim] = refined_obj
        self.prev_obs = bbox3D

        if np.pi / 2.0 < abs(bbox3D[3] - self.avg_angle) < np.pi * 3 / 2.0:
            for r_indx in range(len(self.ref_history)):
                self.ref_history[r_indx][3] = self.fix_alpha(
                    self.ref_history[r_indx][3] + np.pi)

        if self.init_flag:
            self._init_history(refined_obj)
            self.init_flag = False
        else:
            self._update_history(refined_obj)

        self.info = info

    def predict(self, update_state: bool = True):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        with torch.no_grad():
            pred_loc, hidden_pred = self.lstm.predict(
                torch.from_numpy(self.history[..., :self.loc_dim]).view(
                    self.nfr, -1, self.loc_dim).float().to(self.device),
                torch.from_numpy(self.obj_state[:self.loc_dim]).view(
                    -1, self.loc_dim).float().to(self.device),
                self.hidden_pred)

        pred_state = self.obj_state.copy()
        pred_state[:self.loc_dim] = pred_loc.cpu().numpy().flatten()
        pred_state[7:] = pred_state[:3] - self.prev_ref[:3]
        if self.loc_dim > 3:
            pred_state[3] = self.fix_alpha(pred_state[3])

        if update_state:
            self.hidden_pred = hidden_pred
            self.obj_state = pred_state

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return pred_state.flatten()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.obj_state.flatten()

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history
def init_module(layer):
    '''
    Initial modules weights and biases
    '''
    for m in layer.modules():
        if isinstance(m, nn.Conv2d) or \
            isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d) or \
            isinstance(m, nn.GroupNorm):
            m.weight.data.uniform_()
            if m.bias is not None:
                m.bias.data.zero_()


def init_lstm_module(layer):
    '''
    Initial LSTM weights and biases
    '''
    for name, param in layer.named_parameters():

        if 'weight_ih' in name:
            torch.nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)  # initializing the lstm bias with zeros

class LSTM(nn.Module):
    '''
    Estimating object location in world coordinates
    Prediction LSTM:
        Input: 5 frames velocity
        Output: Next frame location
    Updating LSTM:
        Input: predicted location and observed location
        Output: Refined location
    '''

    def __init__(self,
                 batch_size: int,
                 feature_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 loc_dim: int,
                 dropout: float = 0.0):
        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loc_dim = loc_dim

        self.loc2feat = nn.Linear(
            loc_dim,
            feature_dim,
        )

        self.pred2vel = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )

        self.vel2feat = nn.Linear(
            loc_dim,
            feature_dim,
        )

        self.pred_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.refine_lstm = nn.LSTM(
            input_size=2 * feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self._init_param()

    def init_hidden(self, device):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers, self.batch_size,
                            self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.batch_size,
                            self.hidden_size).to(device))

    def predict(self, velocity, location, hc_0):
        '''
        Predict location at t+1 using updated location at t
        Input:
            velocity: (num_seq, num_batch, loc_dim), location from previous update
            location: (num_batch, loc_dim), location from previous update
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            embed: (num_seq, num_batch x feature_dim), location feature
            out: (num_seq x num_batch x hidden_size), lstm output
            merge_feat: (num_batch x hidden_size), the predicted residual
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_seq, num_batch, _ = velocity.shape

        # Embed feature to hidden_size
        embed = self.vel2feat(velocity).view(num_seq, num_batch,
                                             self.feature_dim)

        out, (h_n, c_n) = self.pred_lstm(embed, hc_0)

        # Merge embed feature with output
        # merge_feat = h_n + embed
        merge_feat = out[-1]

        output_pred = self.pred2vel(merge_feat).view(num_batch,
                                                     self.loc_dim) + location

        return output_pred, (h_n, c_n)

    def refine(self, location: torch.Tensor, observation: torch.Tensor,
               prev_location: torch.Tensor, confidence: torch.Tensor, hc_0):
        '''
        Refine predicted location using single frame estimation at t+1
        Input:
            location: (num_batch x 3), location from prediction
            observation: (num_batch x 3), location from single frame estimation
            confidence: (num_batch X 1), depth estimation confidence
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            loc_embed: (1, num_batch x feature_dim), predicted location feature
            obs_embed: (1, num_batch x feature_dim), single frame location feature
            embed: (1, num_batch x 2*feature_dim), location feature
            out: (1 x num_batch x hidden_size), lstm output
            merge_feat: same as out
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_batch = location.shape[0]

        # Embed feature to hidden_size
        loc_embed = self.loc2feat(location).view(num_batch, self.feature_dim)
        obs_embed = self.loc2feat(observation).view(num_batch,
                                                    self.feature_dim)
        embed = torch.cat([loc_embed, obs_embed],
                          dim=1).view(1, num_batch, 2 * self.feature_dim)

        out, (h_n, c_n) = self.refine_lstm(embed, hc_0)

        # Merge embed feature with output
        # merge_feat = h_n + embed
        merge_feat = out

        output_pred = self.pred2vel(merge_feat).view(
            num_batch, self.loc_dim) + observation

        return output_pred, (h_n, c_n)

    def _init_param(self):
        init_module(self.loc2feat)
        init_module(self.vel2feat)
        init_module(self.pred2vel)
        init_lstm_module(self.pred_lstm)
        init_lstm_module(self.refine_lstm)

class VeloLSTM(LSTM):
    '''
    Estimating object location in world coordinates
    Prediction LSTM:
        Input: 5 frames velocity
        Output: Next frame location
    Updating LSTM:
        Input: predicted location and observed location
        Output: Refined location
    '''

    def __init__(self,
                 batch_size: int,
                 feature_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 loc_dim: int,
                 dropout: float = 0.0):
        super(VeloLSTM, self).__init__(batch_size, feature_dim, hidden_size,
                                       num_layers, loc_dim, dropout)
        self.refine_lstm = nn.LSTM(
            input_size=3 * feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self._init_param()

        self.pred2atten = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )
        self.conf2atten = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )
        self.conf2feat = nn.Linear(
            1,
            feature_dim,
            bias=False,
        )
        init_module(self.pred2atten)
        init_module(self.conf2feat)

    def refine(
        self, location: torch.Tensor, observation: torch.Tensor,
        prev_location: torch.Tensor, confidence: torch.Tensor,
        hc_0: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Refine predicted location using single frame estimation at t+1
        Input:
            location: (num_batch x loc_dim), location from prediction
            observation: (num_batch x loc_dim), location from single frame estimation
            prev_location: (num_batch x loc_dim), refined location
            confidence: (num_batch X 1), depth estimation confidence
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            loc_embed: (1, num_batch x feature_dim), predicted location feature
            obs_embed: (1, num_batch x feature_dim), single frame location feature
            conf_embed: (1, num_batch x feature_dim), depth estimation confidence feature
            embed: (1, num_batch x 2*feature_dim), location feature
            out: (1 x num_batch x hidden_size), lstm output
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_batch = location.shape[0]

        pred_vel = location - prev_location
        obsv_vel = observation - prev_location

        # Embed feature to hidden_size
        loc_embed = self.vel2feat(pred_vel).view(num_batch, self.feature_dim)
        obs_embed = self.vel2feat(obsv_vel).view(num_batch, self.feature_dim)
        conf_embed = self.conf2feat(confidence).view(num_batch,
                                                     self.feature_dim)
        embed = torch.cat([
            loc_embed,
            obs_embed,
            conf_embed,
        ], dim=1).view(1, num_batch, 3 * self.feature_dim)

        out, (h_n, c_n) = self.refine_lstm(embed, hc_0)

        delta_vel_atten = torch.sigmoid(self.conf2atten(out)).view(
            num_batch, self.loc_dim)

        output_pred = delta_vel_atten * obsv_vel + (
            1.0 - delta_vel_atten) * pred_vel + prev_location

        return output_pred, (h_n, c_n)

    def predict(
        self, vel_history: torch.Tensor, location: torch.Tensor,
        hc_0: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Predict location at t+1 using updated location at t
        Input:
            vel_history: (num_seq, num_batch, loc_dim), velocity from previous num_seq updates
            location: (num_batch, loc_dim), location from previous update
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            embed: (num_seq, num_batch x feature_dim), location feature
            out: (num_seq x num_batch x hidden_size), lstm output
            attention_logit: (num_seq x num_batch x loc_dim), the predicted residual
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_seq, num_batch, _ = vel_history.shape

        # Embed feature to hidden_size
        embed = self.vel2feat(vel_history).view(num_seq, num_batch,
                                                self.feature_dim)

        out, (h_n, c_n) = self.pred_lstm(embed, hc_0)

        attention_logit = self.pred2atten(out).view(num_seq, num_batch,
                                                    self.loc_dim)
        attention = torch.softmax(attention_logit, dim=0)

        output_pred = torch.sum(attention * vel_history, dim=0) + location

        return output_pred, (h_n, c_n)

def load_checkpoint(model, ckpt_path, optimizer=None, is_test=False):
    global best_score
    assert os.path.isfile(ckpt_path), (
        "No checkpoint found at '{}'".format(ckpt_path))
    print("=> Loading checkpoint '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    if 'best_score' in checkpoint:
        best_score = checkpoint['best_score']
    if 'optimizer' in checkpoint and optimizer is not None:
        print("=> Loading optimizer state")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except (ValueError) as ke:
            print("Cannot load full model: {}".format(ke))
            if is_test: raise ke

    state = model.state_dict()
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except (RuntimeError, KeyError) as ke:
        print("Cannot load full model: {}".format(ke))
        if is_test: raise ke
        state.update(checkpoint['state_dict'])
        model.load_state_dict(state)
    print("=> Successfully loaded checkpoint '{}' (epoch {})".format(
        ckpt_path, checkpoint['epoch']))
    del checkpoint
    torch.cuda.empty_cache()

class TrackletManager:
    def __init__(self,model=None):
        self.tracklets = []
        if model == None:
            self.tracking_model = KalmanBox3DTracker
        else:
            self.tracking_model = lambda box,info : LSTM3DTracker("cuda",model,box,info)
    def plot_bounding_boxes(self,boxes_color):
        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define colors for the bounding boxes
        colors = []
        boxes = []
        for b in boxes_color:
            print(b)
            boxes.append(b[0])
            colors.append(b[1])

        # Loop over the bounding boxes and their respective colors
        for vertices, color in zip(boxes, colors):
            # Extract x, y, and z coordinates
            x = vertices[:, 0]
            y = vertices[:, 1]
            z = vertices[:, 2]

            # Plot the corners as scatter points
            ax.scatter(x, y, z, c=color, marker='o')

            # Define the faces of the bounding box
            faces = np.array([
                [0, 1, 2, 3],  # Bottom face
                [4, 5, 6, 7],  # Top face
                [0, 1, 5, 4],  # Side face 1
                [1, 2, 6, 5],  # Side face 2
                [2, 3, 7, 6],  # Side face 3
                [3, 0, 4, 7]   # Side face 4
            ])

            # Create a Poly3DCollection object and add it to the plot
            collection = Poly3DCollection(vertices[faces])
            collection.set_alpha(0.2)  # Set the transparency of the bounding box
            collection.set_facecolor(color)  # Set the color of the bounding box
            ax.add_collection3d(collection)

            # Define the lines of the bounding box
            lines = [
                [vertices[0], vertices[1]],
                [vertices[1], vertices[2]],
                [vertices[2], vertices[3]],
                [vertices[3], vertices[0]],
                [vertices[4], vertices[5]],
                [vertices[5], vertices[6]],
                [vertices[6], vertices[7]],
                [vertices[7], vertices[4]],
                [vertices[0], vertices[4]],
                [vertices[1], vertices[5]],
                [vertices[2], vertices[6]],
                [vertices[3], vertices[7]]
            ]

            # Create a Line3DCollection object and add it to the plot
            lines_collection = Line3DCollection(lines, colors=color)
            ax.add_collection(lines_collection)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Bounding Boxes')

        # Set axis limits if needed
        # ax.set_xlim([xmin, xmax])
        # ax.set_ylim([ymin, ymax])
        # ax.set_zlim([zmin, zmax])

        # Show the plot
        plt.axis('scaled')

        ax.view_init(-70,-70,-20)
        plt.savefig("dummy_name.png")
    def convertBoxKalman(self,box):
        x,y,z,w,l,h,roty = box
        return np.array([x,y,z,roty,l,w,h])
    def convertBoxTracklet(self,box):
        x,y,z,roty,l,w,h = box
        return np.array([x,y,z,w,l,h,roty])
    def removeDupplicateIou(self,frame):
        try:
            boxes = frame["boxes_3d"]
            iou_matrix = self.compute_iou_matrix(boxes, boxes)
        except Exception as e:
            print(e)
            return frame 
        # print(iou_matrix)
        iou_matrix_upper = np.triu_indices_from(iou_matrix)
        
        index_to_remove = []
        for i,j in zip(*iou_matrix_upper):
            if i != j and iou_matrix[i][j] > 0.5 and frame["labels_3d"][i] == 2:
                index_to_remove.append(i)
            elif i != j and iou_matrix[i][j] > 0.2 and frame["labels_3d"][i] != 2:
                index_to_remove.append(i)

        # print(index_to_remove)
        frame["boxes_3d"] = np.delete(frame["boxes_3d"], index_to_remove, axis=0)      
        frame["labels_3d"] = np.delete(frame["labels_3d"], index_to_remove, axis=0)      
        return frame
    def compute_iou_matrix(self,detections_t, detections_tplus1):
        num_detections_t = len(detections_t)
        num_detections_tplus1 = len(detections_tplus1)
        detections_tplus1 = detections_tplus1+np.ones(np.array(detections_tplus1).shape)*0.001
        iou_matrix = np.zeros((num_detections_t, num_detections_tplus1))

        for i in range(num_detections_t):
            x1, y1, z1, w1, h1, l1, theta1 = detections_t[i]
            corners_3d_box1  = get_3d_box((w1, h1, l1), theta1, (x1, y1, z1))
        
            for j in range(num_detections_tplus1):
                x2, y2, z2, w2, h2, l2, theta2 = detections_tplus1[j]
                corners_3d_box2 = get_3d_box((w2, h2, l2), theta2, (x2, y2, z2))
                # self.plot_bounding_boxes([corners_3d_box1,corners_3d_box2])
                iou_matrix[i, j], _ = box3d_iou(corners_3d_box1,corners_3d_box2)

        return iou_matrix
    def createATrackletModel(self,box_3d):
        return self.tracking_model(self.convertBoxKalman(box_3d),None)
    def tracklets_detections(self, predicted_box, current_detections, row_indices, col_indices, iou_matrix, threshold):

        # for association in list(zip(row_indices, col_indices)):
        for tracklet_id in range(len(self.tracklets)):
            if tracklet_id in row_indices:
                associated_detection_id = col_indices[list(row_indices).index(tracklet_id)]
            # # tracklet_id = association[0]
            # associated_detection_id = association[1]
                if iou_matrix[tracklet_id][associated_detection_id] > threshold:       
                    self.tracklets[tracklet_id]['Kalman'].update(self.convertBoxKalman(current_detections["boxes_3d"][associated_detection_id]))
                    box = self.convertBoxTracklet(self.tracklets[tracklet_id]['Kalman'].get_state()[:7])
                    self.tracklets[tracklet_id]["boxes_3d"] = box
                    self.tracklets[tracklet_id]["labels_3d"] = (current_detections["labels_3d"][associated_detection_id])
                    self.tracklets[tracklet_id]["Timeout"] = 0
                else:
                    self.tracklets[tracklet_id]["Timeout"] += 1
                    self.tracklets[tracklet_id]["boxes_3d"] = predicted_box["boxes_3d"][tracklet_id]
            else:
                self.tracklets[tracklet_id]["Timeout"] += 1
                self.tracklets[tracklet_id]["boxes_3d"] = predicted_box["boxes_3d"][tracklet_id] 
        indexes_to_delete = []
        for index, element in enumerate(self.tracklets):
            if element["Timeout"] > 3:
                indexes_to_delete.append(index)
        for index in sorted(indexes_to_delete, reverse=True):  
            del self.tracklets[index]

        for detection_id in list(set(range(len(current_detections['boxes_3d']))) - set(col_indices)): 
            print("new tracklet",detection_id)
            box_3d = current_detections['boxes_3d'][detection_id]
            self.tracklets.append({
                "boxes_3d":box_3d,
                "labels_3d":current_detections['labels_3d'][detection_id],
                "Timeout":0,
                "Kalman":self.createATrackletModel(box_3d)
            })

        return self.tracklets

    def perform_hungarian_algorithm(self,iou_matrix):
        # Invert the IOU matrix to convert it to a cost matrix
        cost_matrix = 1 - iou_matrix
        # print(cost_matrix)

        # Solve the assignment problem using the Hungarian Algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        return row_indices, col_indices

    def get_last_tracklets(self):
        last_tracklets = []
        for tracklet in self.tracklets:
            last_tracklets.append(tracklet["boxes_3d"])
        return last_tracklets
    def update_tracklets(self,predicted_box,new_detections):


        # COMPUTE IOU MATRIX BETWEEN CURRENT TRACKLETS AND NEW DETECTIONS
        if predicted_box['boxes_3d'] == []:
            for i in range(len(new_detections['boxes_3d'])):
                box_3d = new_detections["boxes_3d"][i]
                self.tracklets.append({
                    "boxes_3d":box_3d,
                    "labels_3d":new_detections["labels_3d"][i],
                    "Timeout":0,
                    "Kalman":self.createATrackletModel(box_3d)
                })
            return self.tracklets
        
        iou_matrix = self.compute_iou_matrix(predicted_box["boxes_3d"],new_detections["boxes_3d"])

        # print(iou_matrix)
        row_indices, col_indices = self.perform_hungarian_algorithm(iou_matrix)

        self.tracklets_detections(predicted_box, new_detections, row_indices, col_indices, iou_matrix, threshold=0.2)
        return self.tracklets
    def predict_kalman(self):
        results={"boxes_3d":[],"labels_3d":[]}
        for tracklet in self.tracklets:
            tracklet['Kalman'].predict()
            box = self.convertBoxTracklet(tracklet['Kalman'].get_state()[:7])
            tracklet['boxes_3d'] = box
            results["boxes_3d"].append(box)
            results["labels_3d"].append(tracklet["labels_3d"])
        return results
    def extract_result_from_tracklets(self):
        results={"boxes_3d":[],"labels_3d":[]}
        for tracklet in self.tracklets:
            results["boxes_3d"].append(tracklet["boxes_3d"])
            results["labels_3d"].append(tracklet["labels_3d"])
    
        results["boxes_3d"] = torch.from_numpy(np.array(results["boxes_3d"]))
        results['labels_3d'] = torch.from_numpy(np.array(results["labels_3d"]))
        return results
    

# Test function
if __name__ == '__main__':
    model = VeloLSTM(
            batch_size=1,
            feature_dim=64,
            hidden_size=128,
            num_layers=2,
            loc_dim=7,
            dropout=0.0).to("cuda")
    model.eval()
    load_checkpoint(
        model,
        "/home/vince/vincent-ml/tools/batch8_min10_seq10_dim7_train_dla34_regress_pretrain_VeloLSTM_kitti_100_linear.pth",
        optimizer=None,
        is_test=True)
    
    trackletmanager = TrackletManager(model)
    refined_frames = []
    # tracketls = trackletmanager.update_tracklets(itemData[0])

    for frame in itemData:
        # boxes = []
        filtered_detections_frame = trackletmanager.removeDupplicateIou(frame)
        predicted_box = trackletmanager.predict_kalman()
        tracketls = trackletmanager.update_tracklets(predicted_box,filtered_detections_frame)
        print("length",len(trackletmanager.tracklets))
        refined_frame = trackletmanager.extract_result_from_tracklets()
        #  = trackletmanager.tracklets[0]['Kalman'].get_state()[:7]
        # boxes.append([get_3d_box((w, h, l), roty, (x, y, z)),'orange'])
        # results_predict = trackletmanager.predict_kalman()
        # x,y,z,roty,w,h,l = trackletmanager.tracklets[0]['Kalman'].get_state()[:7]
        # boxes.append([get_3d_box((w, h, l), roty, (x, y, z)),'b'])
        # tracketls = trackletmanager.update_tracklets(frame)
        # x1, y1, z1, w1, h1, l1, theta1 = trackletmanager.tracklets[0]['boxes_3d']
        # boxes.append([get_3d_box((w1, l1, h1), theta1, (x1, y1, z1)),'g'])
        for tracklet in trackletmanager.tracklets:
            # print("box",tracklet["boxes_3d"][0]+tracklet["boxes_3d"][1]+tracklet["boxes_3d"][2])
            # print("box",tracklet["boxes_3d"][3]+tracklet["boxes_3d"][4]+tracklet["boxes_3d"][5])
            break
        print("------")
        # print(filtered_detections_frame["boxes_3d"][0])
        # print(trackletmanager.tracklets[0]["boxes_3d"])
        # refined_frame = trackletmanager.update_kalman()
        refined_frames.append({"img_bbox":refined_frame,"img_bbox2d":[]})
        # x,y,z,roty,w,h,l = trackletmanager.tracklets[0]['Kalman'].get_state()[:7]
        # boxes.append([get_3d_box((w, h, l), roty, (x, y, z)),'r'])
        # trackletmanager.plot_bounding_boxes(boxes)
    # for frame in itemData:
    #     new_detections = list(frame).copy()
    #     # print(new_detections)

    #     # for detection in new_detections:   
    #     #     x1, y1, z1, w1, h1, l1, theta1 = detection
    #     #     boxes.append(get_3d_box((w1, l1, h1), theta1, (x1, y1, z1)))
    #     # print(len(boxes))
    #     # trackletmanager.plot_bounding_boxes(boxes,'b')
    #     tracketls = trackletmanager.update_tracklets(frame)
    #     refined_frame = trackletmanager.predict_and_update_kalman()
    #     refined_frames.append({"img_bbox":refined_frame,"img_bbox2d":[]})
    #     # print(len(tracketls[0]['3d_boxes']))
    print(refined_frames)
    from dataset.kitti_raw_dataset import KITTIRawDataset

    dataset = KITTIRawDataset("/home/vince/datasets/KITTI/Inference_test/2011_09_26/2011_09_26_drive_0095_sync/image_00/data","/home/vince/datasets/KITTI/Inference_test/2011_09_26/calib_cam_to_cam.txt")
    visualizer = Visualizer(dataset, vis_format=refined_frames)

    visualizer.export_as_video("/home/vince/vincent-ml/logs/monocon_normal_200_v3_fix_dataset", plot_items=['3d','bev'], fps=10)