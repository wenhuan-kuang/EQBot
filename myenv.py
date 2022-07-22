#from gym import spaces, core
import math
import random
from scipy.stats import norm
import scipy.io as sio
import os.path
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class Location(gym.Env):
	def __init__(self):
                self.f = 1
                self.dt = 0.05
                self.win = 51.2
                self.mr = 64
                self.evxmin = -119
                self.evxmax = -117
                self.evymin = 33.5
                self.evymax = 34.5
                self.evzmin = 1
                self.evzmax = 20

                self.action_space = spaces.Discrete(6) 
                self.observation_space =  spaces.Box(low=-1, high=1, shape=(self.mr * 2 + 3,int(self.win/self.dt))) 
                
                # load in data (waveform, event, station)
                self.amp_3d,self.evxs,self.evys,self.evzs,self.evids,self.mask,self.rxs,self.rys = \
                          self.load_data(mypath='./demo_data',\
                          sgyf1=2,sgyt1=2,step1=1,shuffle='False')
                # load in traveltime table
                self.tt_tp = self.load_3Dttfile(mypath='./demo_data')
                # for testing
                self.event_num = 0 -1   # -1 for accounting for multiple reset in load and_run.py
		# 
	
	def reset(self):               
               
               # True location
               self.id_true = [1] # use event 2 (the second) for movie
               #self.id_true =np.array([self.event_num])
               self.event_num = self.event_num + 1

               self.x_true = self.evxs[self.id_true]
               self.y_true = self.evys[self.id_true]
               self.z_true = self.evzs[self.id_true]
               #print('repnum',self.repnum)
               #print('event info:',self.id_true,self.x_true,self.y_true,self.z_true)

               self.count = 0
               self.amp_true = self.amp_3d[self.id_true]
               self.amp_true = np.squeeze(self.amp_true,axis=0)
               #print('nan found:',np.isnan(self.amp_true).any(),np.where(np.isnan(self.amp_true)),self.evids[self.id_true])
               #print(self.amp_true[34]) 
               #print(self.amp_true.shape)
               # Initial location
               #id_x_ini = np.random.uniform(low=0, high=181, size=(1,))
               #id_y_ini = np.random.uniform(low=0, high=81, size=(1,))
               #self.x_ini = -118.9 + np.mean(id_x_ini)*0.01
               #self.y_ini = 33.6 + np.mean(id_y_ini)*0.01
               self.x_ini = -118
               self.y_ini = 34
               self.z_ini = 10
               #self.x_ini = np.mean(self.x_true)  # convert one-element list to variable
               #self.y_ini = np.mean(self.y_true)
               #self.z_ini = np.mean(self.z_true)
               #print('initial:',self.x_ini, self.y_ini, self.z_ini)

               self.x_new = self.x_ini
               self.y_new = self.y_ini
               self.z_new = self.z_ini
               self.step_size_x = 0.25 # degree
               self.step_size_y = 0.25
               self.step_size_z = 2.5 # km

               id_randshift = np.random.uniform(low=0, high=81, size=(1,))
               #self.randshift = np.mean( (id_randshift-40)*self.dt ) + 0 # add rand timeshift with maximum of 2 secs
               self.randshift = 0
               #print('randshift',self.randshift)

               tp = self.get_3Dtp(self.x_ini,self.y_ini,self.z_ini)
               ts = self.get_3Dts(self.x_ini,self.y_ini,self.z_ini)
               #print('tp:',np.array(tp).shape)
               amp1 = [self.myshift(self.amp_true[i], - math.floor((tp[i] - self.randshift)/self.dt)) for i in range(self.mr)]
               amp1 = [tmp[0:int(self.win/self.dt)] for tmp in amp1]
               amp2 = [self.myshift(self.amp_true[i], - math.floor((ts[i] - self.randshift)/self.dt)) for i in range(self.mr)]
               amp2 = [tmp[0:int(self.win/self.dt)] for tmp in amp2]
               amp = np.vstack((amp1,amp2))
               #print('amp:', np.array(amp).shape)
               nsample = np.array(amp).shape[1]
               #print(nsample)
               #print(np.vstack((np.repeat(x_ini,nsample),np.repeat(y_ini,nsample))))
               x_ini_gauss = self.mygauss(u=(self.x_ini-self.evxmin)/(self.evxmax-self.evxmin),\
                       sig=0.02/(self.evxmax-self.evxmin),nsamp=nsample)
               y_ini_gauss = self.mygauss(u=(self.y_ini-self.evymin)/(self.evymax-self.evymin),\
                       sig=0.01/(self.evymax-self.evymin),nsamp=nsample)
               z_ini_gauss = self.mygauss(u=(self.z_ini-self.evzmin)/(self.evzmax-self.evzmin),\
                       sig=0.2/(self.evzmax-self.evzmin),nsamp=nsample)
               #self.state = np.vstack((amp, np.vstack((x_ini_gauss, y_ini_gauss ))))
               self.state = np.vstack((amp, np.vstack((x_ini_gauss, y_ini_gauss, z_ini_gauss))))
               #print('in env', np.array(self.state).shape)
               #print('argmax',np.argmax(x_ini_gauss))
               #print('initial x', (self.evxmax-self.evxmin)*(np.argmax(x_ini_gauss)+1)/nsample + self.evxmin)
               #print(self.state[-2:])
               #self.state = np.vstack((amp, np.vstack((np.repeat(x_ini,nsample),np.repeat(y_ini,nsample)))))
               #self.state = np.append(dtp - np.mean(dtp), np.append(x_ini,y_ini))
               return self.state
	
	def step(self, action):
               self.count = self.count + 1
               #x_old = self.state[-2][-1]
               #y_old = self.state[-1][-1]
               
               #self.step_size = 5/(np.log(self.count)+1)
               if(action==0):
                   self.x_new = self.x_new - self.step_size_x
               elif(action==1):
                   self.x_new = self.x_new + self.step_size_x
               elif(action==2):
                   self.y_new = self.y_new - self.step_size_y
               elif(action==3):
                   self.y_new = self.y_new + self.step_size_y
               elif(action==4):
                   self.z_new = self.z_new - self.step_size_z
               elif(action==5):
                   self.z_new = self.z_new + self.step_size_z
               #self.z_new = np.mean(self.z_true)
               
               dd_new_xy = self.get_dist(self.x_new, self.y_new, self.x_true, self.y_true)
               dd_new_z = np.fabs(self.z_new-self.z_true)
               dd_new = np.sqrt( np.square(dd_new_xy) + np.square(dd_new_z*1.0) )
               #dd_new = np.mean(dd_new)

               reward_gauss = self.mygaussf(u=0,sig=5,dist=dd_new) # 2 or 5 km
               reward = reward_gauss * 0.1 # for mitigating spasity of rewards
               #print('dist xy z 3d = ', dd_new_xy,dd_new_z,dd_new, 'reward=',reward)
               #reward = 0 # -0.01 for optimizing shotest path
               dist_good = 1
               if(dd_new <= dist_good):
                   reward = 1

               done = bool(self.count>=50)
               #done = bool(self.x_new<-119 or self.x_new>-117 or self.y_new<33.5 or self.y_new>34.5 \
               #        or self.z_new<1 or self.z_new>20 or dd_new <= dist_good)

               #print('count = ', self.count,reward,self.x_new,self.y_new,self.z_new)
               if(self.x_new<-119 or self.x_new>-117 or self.y_new<33.5 or self.y_new>34.5):
                   reward = reward - 0.1
               if(self.z_new<self.evzmin or self.z_new>self.evzmax):
                   reward = reward - 0.1

               if(self.x_new<-119):
                   self.x_new = self.x_new + self.step_size_x * 1
               if(self.x_new>-117):
                   self.x_new = self.x_new - self.step_size_x * 1
               if(self.y_new<33.5):
                   self.y_new = self.y_new + self.step_size_y * 1
               if(self.y_new>34.5):
                   self.y_new = self.y_new - self.step_size_y * 1
               if(self.z_new<self.evzmin):
                   self.z_new = self.z_new + self.step_size_z * 1
               if(self.z_new>self.evzmax):
                   self.z_new = self.z_new - self.step_size_z * 1
               
               self.step_size_x = self.step_size_x * 0.9
               self.step_size_y = self.step_size_y * 0.9
               self.step_size_z = self.step_size_z * 0.9
               step_min = 0.002 # degree
               step_min_z = 0.5 # km
               if(self.step_size_x < step_min or self.step_size_y < step_min):
                   self.step_size_x = step_min
                   self.step_size_y = step_min
               if(self.step_size_z < step_min_z):
                   self.step_size_z = step_min_z

               print('step count = ', self.count)
               #print('count = ', self.id_true,self.count,reward,dd_new_xy,dd_new_z,dd_new,\
               #        self.x_new,self.y_new,self.z_new, \
               #        self.step_size_x,self.step_size_y,self.step_size_z,done)
               #if(self.count%10==0): 
                   #print('count = ', self.count,reward,dd_new,self.x_new,self.y_new,self.z_new, done)

               tp_new = self.get_3Dtp(self.x_new,self.y_new,self.z_new)
               ts_new = self.get_3Dts(self.x_new,self.y_new,self.z_new)
               #print('tp:',np.array(tp).shape)
               amp1 = [self.myshift(self.amp_true[i], - math.floor((tp_new[i] - self.randshift)/self.dt)) for i in range(self.mr)]
               amp1 = [tmp[0:int(self.win/self.dt)] for tmp in amp1]
               amp2 = [self.myshift(self.amp_true[i], - math.floor((ts_new[i] - self.randshift)/self.dt)) for i in range(self.mr)]
               amp2 = [tmp[0:int(self.win/self.dt)] for tmp in amp2]
               amp = np.vstack((amp1,amp2))
               nsample = np.array(amp).shape[1]
               x_new_gauss = self.mygauss(u=(self.x_new-self.evxmin)/(self.evxmax-self.evxmin),\
                       sig=0.02/(self.evxmax-self.evxmin),nsamp=nsample)
               y_new_gauss = self.mygauss(u=(self.y_new-self.evymin)/(self.evymax-self.evymin),\
                       sig=0.01/(self.evymax-self.evymin),nsamp=nsample)
               z_new_gauss = self.mygauss(u=(self.z_new-self.evzmin)/(self.evzmax-self.evzmin),\
                       sig=0.2/(self.evzmax-self.evzmin),nsamp=nsample)
               self.state = np.vstack((amp, np.vstack((x_new_gauss, y_new_gauss, z_new_gauss ))))
               #print('argmax',np.argmax(x_ini_gauss))
               #print('env x', (self.evxmax-self.evxmin)*( np.argmax(x_new_gauss) )/(nsample-1) + self.evxmin)
               #print(nsample)
               #print(np.vstack((np.repeat(x_ini,nsample),np.repeat(y_ini,nsample))))
               #self.state = np.vstack((amp, np.vstack((np.repeat(x_new,nsample),np.repeat(y_new,nsample)))))
               #print(np.array(self.state).shape)
               #self.state = np.append(dtp - np.mean(dtp), np.append(x_new,y_new))
               
               info = {} # 
               #print('repeat:',self.repeat)
               #print(action)
               #print(self.state)
               #print(reward)
               #print(done)
               #reward = reward/100
               return self.state, reward, done, info
	
	# 
	def myshift(self,amp,k):
                   npts = len(amp)
                   #amp_new = np.hstack(amp[npts-k:],amp[:npts-k])
                   if(k>0):
                        amp_new = np.hstack((np.zeros_like(amp[npts-k:]), amp[:npts-k]))
                   else:
                        amp_new = np.hstack((amp[-k:], np.zeros_like(amp[:-k])))
                   return amp_new

	def get_dist(self,x1,y1,x2,y2):
               dist = self.geodistance(x1,y1,x2,y2)
               return dist

	def get_tp(self,x,y,z):
               tp = []
               for rx,ry in zip(self.rxs,self.rys):
                   dist = self.geodistance(x,y,rx,ry)
                   pdx=1
                   pdz=1
                   ipx=np.round(dist/pdx)+1
                   ipz=np.round(z/pdz)+1
                   ipx=ipx.astype(int)
                   ipz=ipz.astype(int)
                   #print(x,y,rx,ry)
                   #print(dist,ipx,ipz)
                   tp_one=self.tt_tp[ipz][ipx]*1.00;
                   tp.append(tp_one)
               return tp

	def get_3Dtp(self,x,y,z):
               tp = []
               for ir in range(self.mr):
                   distx = x - self.evxmin
                   disty = y - self.evymin
                   pdx=0.01
                   pdy=0.01
                   pdz=1
                   ipx=np.round(distx/pdx)
                   ipy=np.round(disty/pdy)
                   ipz=np.round(z/pdz)
                   ipx=ipx.astype(int)
                   ipy=ipy.astype(int)
                   ipz=ipz.astype(int)
                   if ipx>200:
                      ipx=200
                   if ipy>100:
                      ipy=100
                   if ipz>49:
                      ipz=49
                   #print('xyz',distx,disty,z)
                   #print('in get_3Dtp',ir,ipx,ipy,ipz)
                   tp_one=self.tt_tp[ir][ipx][ipy][ipz];
                   tp.append(tp_one)
               return tp

	def get_3Dts(self,x,y,z):
               ts = []
               for ir in range(self.mr):
                   distx = x - self.evxmin
                   disty = y - self.evymin
                   pdx=0.01
                   pdy=0.01
                   pdz=1
                   ipx=np.round(distx/pdx)
                   ipy=np.round(disty/pdy)
                   ipz=np.round(z/pdz)
                   ipx=ipx.astype(int)
                   ipy=ipy.astype(int)
                   ipz=ipz.astype(int)
                   if ipx>200:
                      ipx=200
                   if ipy>100:
                      ipy=100
                   if ipz>49:
                      ipz=49
                   #print('xyz',distx,disty,z)
                   #print('in get_3Dtp',ir,ipx,ipy,ipz)
                   ts_one=self.tt_tp[ir][ipx][ipy][ipz] * 1.7385;
                   ts.append(ts_one)
               return ts

	def get_ts(self,x,y,z):
               ts = []
               for rx,ry in zip(self.rxs,self.rys):
                   dist = self.geodistance(x,y,rx,ry)
                   pdx=1
                   pdz=1
                   ipx=np.round(dist/pdx)+1
                   ipz=np.round(z/pdz)+1
                   ipx=ipx.astype(int)
                   ipz=ipz.astype(int)
                   #print(x,y,rx,ry)
                   #print(dist,ipx,ipz)
                   ts_one=self.tt_ts[ipz][ipx]*1.00;
                   ts.append(ts_one)
               return ts
               #dist = ((x-np.array(self.rxs))**2 + (y-np.array(self.rys))**2)**0.5
               #ts = dist/3.0
               #return ts

	def ricker(self,f,dt,win):
               number = win/dt;
               tt = np.array(range(math.floor(-number/2)+1, math.floor(number/2)+1)) * dt
               #print(tt)
               #print(np.pi*f*tt)
               amp = np.multiply((1-2*np.square(np.pi*f*tt)), np.exp(-np.square(np.pi*f*tt)))
               return amp

	def load_data(self,mypath='train_data',sgyf1=1,sgyt1=1,step1=1,shuffle='true'):
               data = []
               evx = []
               evy = []
               evz = []
               evid = []
               mask = []
               rx = []
               ry = []
               for i in range(sgyf1,sgyt1+1,step1):
                    filename="%s/%06d.mat" %(mypath,i)
                    print('filename = %s' %(filename))
                    if os.path.exists(filename):
                       data_mat = sio.loadmat(filename);
                       data_in=data_mat['amp_3d'] #extract variables we need
                       for tmp in data_in:
                           data.append(tmp)
        
                       evx_in=data_mat['evx_true'] #extract variables we need
                       evy_in=data_mat['evy_true'] #extract variables we need
                       evz_in=data_mat['evz_true'] #extract variables we need
                       evid_in=data_mat['evid_new'] #extract variables we need
                       evx_in=np.squeeze(evx_in)
                       evy_in=np.squeeze(evy_in)
                       evz_in=np.squeeze(evz_in)
                       evid_in=np.squeeze(evid_in)
                       for tmpx, tmpy, tmpz, tmpid in zip(evx_in,evy_in,evz_in,evid_in):
                           evx.append(tmpx)
                           evy.append(tmpy)
                           evz.append(tmpz)
                           evid.append(tmpid)

                       rx_in=data_mat['rx'] #extract variables we need
                       ry_in=data_mat['ry'] #extract variables we need
                       rx_in=np.squeeze(rx_in)
                       ry_in=np.squeeze(ry_in)
                       for tmpx, tmpy in zip(rx_in,ry_in):
                           rx.append(tmpx)
                           ry.append(tmpy)
                    else:
                       print('File %s not found' %(filename));
               # read in mask for random probability based on event density
               filename="%s/mask.mat" %(mypath)
               #print('mask file = %s' %(filename))
               if os.path.exists(filename):
                  mask_mat = sio.loadmat(filename);
                  mask_in=mask_mat['mask'] #extract variables we need
                  mask_in=np.squeeze(mask_in)
                  for tmp in mask_in:
                      mask.append(tmp)

               index=[i for i in range(len(evx))]
               random.seed(7)
               if shuffle == 'true':
                   random.shuffle(index)
                   data = [data[i] for i in index]
                   evx = [evx[i] for i in index]
                   evy = [evy[i] for i in index]
                   evz = [evz[i] for i in index]
                   evid = [evid[i] for i in index]
                   mask = [mask[i] for i in index]

               data=np.array(data)
               evx=np.array(evx)
               evy=np.array(evy)
               evz=np.array(evz)
               evid=np.array(evid)
               mask=np.array(mask)
               rx=np.array(rx)
               ry=np.array(ry)
               print('Input data shape:',data.shape)
               #print(evx.shape)
               #print(mask.shape)
               #print(rx.shape)
               #print("read data finished\n")
               #nums=10
               #return data[:nums],evx[:nums],evy[:nums],evz[:nums],evid[:nums],mask[:nums],rx,ry
               return data,evx,evy,evz,evid,mask,rx,ry

	def load_station(self,mypath='train_data'):  # not used
               rx = []
               ry = []
               filename="%s/station.mat" %(mypath)
               #print('filename = %s' %(filename))
               if os.path.exists(filename):
                  data_mat = sio.loadmat(filename);
                  rx_in=data_mat['rx'] #extract variables we need
                  ry_in=data_mat['ry'] #extract variables we need
                  rx_in=np.squeeze(rx_in)
                  ry_in=np.squeeze(ry_in)
                  for tmpx, tmpy in zip(rx_in,ry_in):
                      rx.append(tmpx)
                      ry.append(tmpy)
               else:
                  print('File %s not found' %(filename));
               
               rx=np.array(rx)
               ry=np.array(ry)
               print(rx.shape)
               #print("read station finished\n")
               return rx,ry

	def load_ttfile(self,mypath='train_data',key='5'):  
               tt = []
               filename="%s/table_%s.mat" %(mypath,key)
               #print('filename = %s' %(filename))
               if os.path.exists(filename):
                  data_mat = sio.loadmat(filename);
                  tt_in=data_mat['ttbl'] #extract variables we need
                  for tmp in tt_in:
                      tt.append(tmp)
               else:
                  print('File %s not found' %(filename));
               
               tt=np.array(tt)
               print(tt.shape)
               #print("read traveltime table finished\n")
               return tt

	def load_3Dttfile(self,mypath='train_data'):  
               tt = []
               filename="%s/tp.mat" %(mypath)
               print('3D tpfile = %s' %(filename))
               if os.path.exists(filename):
                  data_mat = sio.loadmat(filename);
                  tt=data_mat['tp'] #extract variables we need
                  #tt_in=data_mat['tp'] #extract variables we need
                  #print(np.array(tt_in).shape)
                  #for tmp in tt_in:
                  #    tt.append(tmp)
               else:
                  print('File %s not found' %(filename));
               
               tt=np.array(tt)
               print('ttfile shape:',tt.shape)
               #print("read traveltime table finished\n")
               return tt

	def geodistance(self,lng1,lat1,lng2,lat2):
               lng1,lat1,lng2,lat2 = map(math.radians,[float(lng1),float(lat1),float(lng2),float(lat2)])
               dlon = lng2-lng1
               dlat = lat2-lat1
               a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
               distance = 2*math.asin(math.sqrt(a))*6371*1000
               distance = round(distance/1000,3)
               return distance

	def mygauss(self,u=0.5,sig=0.2,nsamp=1024): # generate a gauss array
               x = np.linspace(0,1,nsamp)
               y_sig = np.exp(-(x - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig)
               y_sig = y_sig/np.max(y_sig)
               return y_sig

	def mygaussf(self,u=0.5,sig=0.2,dist=10): # input one dist, output one value
               #x = np.linspace(0,1,nsamp)
               y = np.exp(-(dist - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig)
               y_max = np.exp(-(0 - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig)
               y = y/y_max
               y = np.mean(y) # 1 element arry to value
               return y

#	def _get_observation(self, action):
		#return obs
	
#	def _get_reward(self):
		#return reward

#	def _get_done(self):
		#return done
