import numpy as np
from utils import *
from matplotlib import pyplot as plt
import os


# if __name__ == '__main__':
ID = 20
filename = './data/00'+str(ID)+'.npz'
t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

# ## Inspecting the data
# # Linear velocity
# fig, ax = plt.subplots(3,1,figsize=(10,15))
# plt.clf()
# plt.subplot(3,1,1)
# plt.plot(linear_velocity[0,:])
# plt.title('Linear Velocity')
# plt.subplot(3,1,2)
# plt.plot(linear_velocity[1,:])
# plt.subplot(3,1,3)
# plt.plot(linear_velocity[2,:])
# plt.savefig('Linear_Velocity_'+str(ID)+'.png')
# # angular velocity
# fig, ax = plt.subplots(3,1,figsize=(10,15))
# plt.clf()
# plt.subplot(3,1,1)
# plt.plot(rotational_velocity[0,:])
# plt.title('Rotational Velocity')
# plt.subplot(3,1,2)
# plt.plot(rotational_velocity[1,:])
# plt.subplot(3,1,3)
# plt.plot(rotational_velocity[2,:])
# plt.savefig('Rotational_Velocity_'+str(ID)+'.png')


# noise level: 
n1 = 1 # motion noise
n2 = 30 # observation noise

## (a) IMU Localization via EKF Prediction
# world frame: set as IMU frame at t=0
MU = np.zeros((4,4,t.shape[1]))
MU_inv = np.zeros((4,4,t.shape[1])) # IMU in world frame
SIGMA = np.zeros((6,6,t.shape[1]))
W = np.eye(6)*n1
MU[:,:,0] = np.eye(4)
MU_inv[:,:,0] = np.eye(4)
SIGMA[:,:,0] = np.eye(6)
for i in range(0,t.shape[1]-1):
	mu = MU[:,:,i]
	sigma = SIGMA[:,:,i]
	tau = t[0,i+1] - t[0,i] # seconds
	u = np.hstack([linear_velocity[:,i],rotational_velocity[:,i]])
	u_hat = se3_hat(u)
	ks_hat = -tau*u_hat
	ks_exp = se3_exp(ks_hat)
	R = ks_exp[:3,:3]
	p = ks_exp[:3,3]; p_hat = so3_hat(p)
	ks_exp_ad = np.vstack((np.hstack((R,np.matmul(p_hat,R))),np.hstack((np.zeros((3,3)),R))))
	MU[:,:,i+1] = np.matmul(ks_exp,mu)
	MU_inv[:,:,i+1] = T_inv(MU[:,:,i+1])
	SIGMA[:,:,i+1] = np.matmul(np.matmul(ks_exp_ad,sigma),ks_exp_ad.T) + tau**2*W
	print(i)

fig,ax = visualize_trajectory_2d(MU_inv,path_name=str(ID),show_ori=True)
fig.savefig('Prediction_'+str(ID)+'w'+str(n1)+'v'+str(n2)+'.png',dpi=500)

## (b) Landmark Mapping via EKF Update
M = np.hstack((np.vstack((K[:2,:],K[:2,:])),np.array([[0,0,-b*K[0,0],0]]).T)) # intrinsics of left camera
oTi = cam_T_imu
D = np.vstack((np.eye(3),np.zeros((1,3))))
# Initialize mu and sigma
mu_m0 = np.zeros((4,features.shape[1]))
for i in range(0,t.shape[1]-1):
	obs_idx = np.where(features[0,:,i]!=-1)[0]
	for j in obs_idx:
		obs = features[:,j,i]
		s_w = reverse_proj(obs,MU[:,:,i],M,oTi)
		mu_m0[:,j] = s_w[:,0]
		print(j)
	print(i)

sigma_m0 = np.eye(3*features.shape[1])
# Inspecting the initialize map
fig,ax = plt.subplots(figsize=(5,5))
plt.scatter(mu_m0[0,:],mu_m0[1,:],s=3)
ax.axis('equal')
fig.savefig('Initial_Map_'+str(ID)+'w'+str(n1)+'v'+str(n2)+'.png',dpi=500)

# Update the map
i = 0; mu_m = mu_m0; sigma_m = sigma_m0
# MU_M = np.zeros(())
for i in range(0,t.shape[1]-1):
	obs_idx = np.where(features[0,:,i]!=-1)[0]
	Nt = obs_idx.shape[0]
	H = np.zeros((4*Nt,3*features.shape[1]))
	V = np.eye(4*Nt)*n2
	inno = np.zeros((4*Nt))
	k = 0
	for j in obs_idx:
		s_o = np.matmul(np.matmul(oTi,MU[:,:,i]),mu_m[:,j])
		pi_deriv = 1/s_o[2]*np.array([[1,0,-s_o[0]/s_o[2],0],[0,1,-s_o[1]/s_o[2],0],[0,0,0,0],[0,0,-s_o[3]/s_o[2],1]])
		Hij = np.matmul(np.matmul(M,pi_deriv),np.matmul(np.matmul(oTi,MU[:,:,i]),D))
		H[4*k:4*(k+1),3*j:3*(j+1)] = Hij
		inno[4*k:4*(k+1)] = features[:,j,i] - np.matmul(M,s_o/s_o[2])
		k=k+1
	Kt = np.matmul(np.matmul(sigma_m,H.T),np.linalg.inv(np.matmul(np.matmul(H,sigma_m),H.T)+V))
	K_inno = np.matmul(Kt,inno)
	mu_m_delta = np.vstack((K_inno.reshape(features.shape[1],3).T,np.zeros((1,features.shape[1]))))
	mu_m_tp1 = mu_m + mu_m_delta
	sigma_m_tp1 = np.matmul((np.eye(3*features.shape[1])-np.matmul(Kt,H)),sigma_m)
	mu_m = mu_m_tp1; sigma_m = sigma_m_tp1
	print(i)

fig,ax = plt.subplots(figsize=(5,5))
plt.plot(MU_inv[0,3,:],MU_inv[1,3,:],'r-')
plt.scatter(mu_m[0,:],mu_m[1,:],s=3)
ax.axis('equal')
fig.savefig('MapUpdate_'+str(ID)+'w'+str(n1)+'v'+str(n2)+'.png',dpi=500)


## (c) Visual-Inertial SLAM (Extra Credit)
# 1) initialization: location prediction
FN = features.shape[1]
MU = np.zeros((4,4,t.shape[1]))
MU_inv = np.zeros((4,4,t.shape[1])) # IMU in world frame
SIGMA = np.zeros((6,6,t.shape[1]))
W = np.eye(6)*n1
MU[:,:,0] = np.eye(4)
MU_inv[:,:,0] = np.eye(4)
SIGMA[:,:,0] = np.eye(6)
# 2) initialization: map update
M = np.hstack((np.vstack((K[:2,:],K[:2,:])),np.array([[0,0,-b*K[0,0],0]]).T)) # intrinsics of left camera
oTi = cam_T_imu
D = np.vstack((np.eye(3),np.zeros((1,3))))
# i = 0; mu_m = mu_m0; sigma_m = sigma_m0 # initial map from part (b)
i = 0; mu_m = np.zeros((4,FN)); sigma_m = np.eye((3*FN))
mu_c = np.zeros((3*FN+6)); sigma_c = np.eye((3*FN+6))
# Record variables
Record_delta_m = np.zeros((t.shape[1]))
Record_delta = np.zeros((2,t.shape[1])) # update of pose
ex=0 # count the skipped update step
for i in range(0,t.shape[1]-1):
	# 1) location prediction
	mu = MU[:,:,i]
	sigma = SIGMA[:,:,i]
	tau = t[0,i+1] - t[0,i] # seconds
	u = np.hstack([linear_velocity[:,i],rotational_velocity[:,i]])
	u_hat = se3_hat(u)
	ks_hat = -tau*u_hat
	ks_exp = se3_exp(ks_hat)
	R = ks_exp[:3,:3]
	p = ks_exp[:3,3]; p_hat = np.array([[0,-p[2],p[1]],[p[2],0,-p[0]],[-p[1],p[0],0]])
	ks_exp_ad = np.vstack((np.hstack((R,np.matmul(p_hat,R))),np.hstack((np.zeros((3,3)),R))))
	mu_tp1t = np.matmul(ks_exp,mu)
	sigma_tp1t = np.matmul(np.matmul(ks_exp_ad,sigma),ks_exp_ad.T) + tau**2*W
	sigma_c[3*FN:,3*FN:] = sigma_tp1t
	# 2) Combined update
	obs_idx = np.where(features[0,:,i]!=-1)[0]
	if obs_idx.shape[0] == 0: 
		MU[:,:,i+1] = mu_tp1t
		SIGMA[:,:,i+1] = sigma_tp1t
		MU_inv[:,:,i+1] = T_inv(MU[:,:,i+1])
		continue
	Nt = obs_idx.shape[0]
	H_m = np.zeros((4*Nt,3*FN))
	V = np.eye(4*Nt)*n2
	inno_m = np.zeros((4*Nt))
	H_p = np.zeros((4*Nt,6))
	k = 0
	for j in obs_idx:
		if np.count_nonzero(mu_m[:,j]==0)==4: 
			obs = features[:,j,i]
			s_w = reverse_proj(obs,mu_tp1t,M,oTi)
			mu_m[:,j] = s_w[:,0]
		s_o = np.matmul(np.matmul(oTi,mu_tp1t),mu_m[:,j])
		pi_deriv = 1/s_o[2]*np.array([[1,0,-s_o[0]/s_o[2],0],[0,1,-s_o[1]/s_o[2],0],[0,0,0,0],[0,0,-s_o[3]/s_o[2],1]])
		Hij = np.matmul(np.matmul(M,pi_deriv),np.matmul(np.matmul(oTi,mu_tp1t),D))
		H_m[4*k:4*(k+1),3*j:3*(j+1)] = Hij
		mumap = np.matmul(mu_tp1t,mu_m[:,j])
		mumap_c = np.vstack((np.hstack((mumap[3]*np.eye(3),-so3_hat(mumap[:3]))),np.zeros((1,6))))
		Hij = np.matmul(np.matmul(np.matmul(M,pi_deriv),oTi),mumap_c)
		H_p[4*k:4*(k+1),:] = Hij
		inno_m[4*k:4*(k+1)] = features[:,j,i] - np.matmul(M,s_o/s_o[2])
		k=k+1
	# Update map mean
	Kt_m = np.matmul(np.matmul(sigma_m,H_m.T),np.linalg.inv(np.matmul(np.matmul(H_m,sigma_m),H_m.T)+V))
	K_inno = np.matmul(Kt_m,inno_m)
	delta_m = np.vstack((K_inno.reshape(FN,3).T,np.zeros((1,FN))))
	delta_distance = np.sqrt(np.sum(delta_m[:2,:]**2,axis=0))
	Record_delta_m[i]= np.max(delta_distance)
	if np.max(delta_distance)>1: 
		ex_idx = np.where(delta_distance>1)[0]
		delta_m[:,ex_idx] = 0
		# for ex_idx_i in ex_idx: H_m[:,3*ex_idx_i:3*(ex_idx_i+1)] = 0 
	mu_m = mu_m + delta_m
	# Update location mean
	Kt_p = np.matmul(np.matmul(sigma_tp1t,H_p.T),np.linalg.inv(np.matmul(np.matmul(H_p,sigma_tp1t),H_p.T)+V))
	kexi_hat = se3_hat(np.matmul(Kt_p,inno_m))
	kexi_exp = se3_exp(kexi_hat)
	mu_tp1tp1 = np.matmul(kexi_exp, mu_tp1t)
	delta_mu = np.abs(se3_log(mu_tp1tp1) - se3_log(mu_tp1t))
	delta_drift = np.sqrt(np.sum(delta_mu[:3]**2,axis=0))
	delta_angle = np.sqrt(np.sum(delta_mu[3:]**2,axis=0))
	Record_delta[:,i] = np.array([delta_drift,delta_angle])
	if delta_angle>0.005 or delta_drift>50: 
		# MU[:,:,i+1] = mu_tp1t; MU_inv[:,:,i+1] = T_inv(MU[:,:,i+1]); SIGMA[:,:,i+1]=sigma_tp1t
		# print(i);ex=ex+1;continue
		mu_tp1tp1 = np.copy(mu_tp1t); print(i); ex=ex+1
	MU[:,:,i+1] = mu_tp1tp1
	MU_inv[:,:,i+1] = T_inv(MU[:,:,i+1])
	# Update big sigma
	H_c = np.hstack((H_m,H_p))
	Kt_c = np.matmul(np.matmul(sigma_c,H_c.T),np.linalg.inv(np.matmul(np.matmul(H_c,sigma_c),H_c.T)+V))
	sigma_c = np.matmul(np.eye(3*FN+6)-np.matmul(Kt_c,H_c),sigma_c)
	sigma_m = sigma_c[:3*FN,:3*FN]
	sigma_p = sigma_c[3*FN:,3*FN:]
	SIGMA[:,:,i+1] = sigma_p


fig,ax = plt.subplots(figsize=(5,5))
plt.plot(MU_inv[0,3,:],MU_inv[1,3,:],'r-')
plt.scatter(mu_m[0,:],mu_m[1,:],s=3)
ax.axis('equal')
# fig.savefig('FullSLAM_v2_'+str(ID)+'w'+str(n1)+'v'+str(n2)+'.png',dpi=500)
fig.savefig('FullSLAM_v3_'+str(ID)+'w'+str(n1)+'v'+str(n2)+'.png',dpi=500)
# fig.savefig('FullSLAM_'+str(ID)+'.png',dpi=500)

# fig,ax = visualize_trajectory_2d(MU_inv,path_name=str(ID),show_ori=True)

