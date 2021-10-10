import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

def load_data(file_name):
  '''
  function to read visual features, IMU measurements and calibration parameters
  Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
  Output:
      t: time stamp
          with shape 1*t
      features: visual feature point coordinates in stereo images, 
          with shape 4*n*t, where n is number of features
      linear_velocity: IMU measurements in IMU frame
          with shape 3*t
      rotational_velocity: IMU measurements in IMU frame
          with shape 3*t
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
          close to 
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
  '''
  with np.load(file_name) as data:
      t = data["time_stamps"] # time_stamps
      features = data["features"] # 4 x num_features : pixel coordinates of features
      linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
      rotational_velocity = data["rotational_velocity"] # rotational velocity measured in the body frame
      K = data["K"] # intrindic calibration matrix
      b = data["b"] # baseline
      cam_T_imu = data["cam_T_imu"] # Transformation from imu to camera frame
  return t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu


def visualize_trajectory_2d(pose,path_name="Unknown",show_ori=False):
  '''
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose, 
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  '''
  fig,ax = plt.subplots(figsize=(5,5))
  n_pose = pose.shape[2]
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  if show_ori:
      select_ori_index = list(range(0,n_pose,int(n_pose/50)))
      yaw_list = []
      for i in select_ori_index:
          _,_,yaw = mat2euler(pose[:3,:3,i])
          yaw_list.append(yaw)
      dx = np.cos(yaw_list)
      dy = np.sin(yaw_list)
      dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
      ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  # plt.show(block=True)
  return fig, ax


def T_inv(T):
  R_T = T[:3,:3]; p_T=T[:3,3]
  T_inv = np.vstack((np.hstack((R_T.T,-np.matmul(R_T.T,p_T)[:,None])),np.array([[0,0,0,1]])))
  return T_inv

def se3_hat(u):
  u_hat = np.array([[0,-u[5],u[4],u[0]],[u[5],0,-u[3],u[1]],[-u[4],u[3],0,u[2]],[0,0,0,1]])
  return u_hat

def se3_invhat(u_hat):
  u = np.array([u_hat[0,3],u_hat[1,3],u_hat[2,3],u_hat[2,1],u_hat[0,2],u_hat[1,0]])
  return u

def so3_hat(p): 
  p_hat = np.array([[0,-p[2],p[1]],[p[2],0,-p[0]],[-p[1],p[0],0]])
  return p_hat
  

def se3_exp(ks_hat):
  theta_norm = np.sqrt(ks_hat[0,1]**2+ks_hat[0,2]**2+ks_hat[1,2]**2)
  ks_exp = np.eye(4) + ks_hat + (1-np.cos(theta_norm))/(theta_norm**2)*np.matmul(ks_hat,ks_hat) \
  + (theta_norm-np.sin(theta_norm))/(theta_norm**3)*np.matmul(ks_hat,np.matmul(ks_hat,ks_hat))
  return ks_exp

def se3_log(T):
  '''
  INPUT: T: pose in SE(3)
  OUTPUT: kexi 6*1
  '''
  R = T[:3,:3]; p=T[:3,3]; kexi = np.zeros((6))
  if (R==np.eye(3)).all(): kexi[:3] = p
  else:
    theta_norm = np.arccos((np.trace(R)-1)/2)
    theta_hat = theta_norm/(2*np.sin(theta_norm))*(R-R.T)
    theta = np.array([theta_hat[2,1],theta_hat[0,2],theta_hat[1,0]])
    JL_inv = np.eye(3)-0.5*theta_hat+((1+np.cos(theta_norm))/theta_norm**2-1/(2*theta_norm*np.sin(theta_norm)))*np.matmul(theta_hat,theta_hat)
    ro = np.matmul(JL_inv,p)
    kexi[:3] = ro; kexi[3:] = theta
  return kexi


def reverse_proj(obs,Tinv,M,oTi):
  '''
  INPUT: 
    obs: 4*1, stereo camera measurements
    Tinv: 4*4, world in the IMU frame
  OUTPUT:
    s_w: 4*1, map coordinates in the world frame
  '''
  z = -M[2,3]/(obs[0]-obs[2])
  x = (obs[0]-M[0,2])*z/M[0,0]
  y = (obs[1]-M[1,2])*z/M[1,1]
  s_o = np.array([x,y,z,1])[:,None]
  s_w = np.matmul(T_inv(np.matmul(oTi,Tinv)),s_o)
  return s_w



