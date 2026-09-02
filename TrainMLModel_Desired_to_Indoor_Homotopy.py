
from   datetime import timezone
from   datetime import date, datetime, timedelta
import os
import pytz

import numpy as np
import sys
import pickle

from os.path import exists


## Learning parameters
Discount      = 0.98      # Forgetting factor, time basis is day
ExitThreshold = 1e-2      # Relative loss function improvement to exit (1e-2 is used in cronetab for computational time)


#python3 -m venv path/to/venv;

"""
python3 -m venv path/to/venv
source path/to/venv/bin/activate
python3 -m pip install pytz
"""

timez    = pytz.timezone('Europe/Oslo')
Now      = datetime.now().astimezone(timez)


if sys.stdout.isatty():
    # If called from a terminal, enable plotting
    Manual = True
else:
    # If called from crontab, neutralize plotting & save (update) plan
    Manual        = False
    ExitThreshold = 1e-2    # Overwrite the exit threshold for cronetab
    print('-------------- Crontab launch at : '+str(Now)+' --------------')
   

WarmStart = True
ResetL1   = False



Vars = {'Past'   : ['Indoor',   'Supply', 'Return'  ],  #'Supply'
        'Future' : ['Outdoor',  'Feature']} #,'SetTemp', 'Desired'

Target = ['Indoor']


def SoftPlus(x , p):
    if not(isinstance(x, np.ndarray)):
        x = np.array(x)
        
    y = np.log( 1 + np.exp(p * x) ) / p
    return y

print('Load Regression Data')
datatime = datetime.fromtimestamp(os.path.getctime('DataRegression.pkl'))

f = open('DataRegression.pkl',"rb")
Data = pickle.load(f)
f.close()

dt = Data['Sampling']
assert(60/dt == np.round(60/dt))
print('Sampling '+str(dt)+' min')


MSsetup = {'Depth' : int(   6*60 / dt ), # Depth       6h
           'Pred'  : int(  36*60 / dt )} # Prediction 24h
           




##########################

## Extra feature
#SupplyRef       = np.interp(np.array(Data['Outdoor']),HeatCurve['Outdoor'],HeatCurve['Supply'])
#Data['Feature'] = np.maximum(SupplyRef * ( DemandParam['Offset'] + DemandParam['Gain'] * np.array(Data['SetTemp']) )  + DemandParam['Soft0' ], DemandParam['MinDesired'])

#Data['Feature'] = [val-22 for val in Data['Desired']]   # Note: identify from Desired signal to make the modelling invariant against supply curve changes

Data['Feature'] = Data['Desired']

if Manual:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    plt.close('all')
    """
    figID = plt.figure(15,figsize=(6,6))
    #ax1 = figID.add_subplot(211) # High temps

    #ax1.plot(Data['SetTemp'],Data['Desired'],linestyle='none',marker='.')



    ax1 = figID.add_subplot(111) # High temps

    ax1.step(Data['Time'],Data['SetTemp'],label='Set',  color='c', where='post')
    #ax1.step(Data['Time'],Data['Supply'],label='Supply',  color='g', where='post')
    #ax1.step(Data['Time'],SupplyRef,label='Outdoor-->Supply',  color='k', where='post')
    ax1.step(Data['Time'],Data['Feature'],label='Feature',  color='r', where='post',linewidth=3)
    ax1.step(Data['Time'],Data['Desired'],label='Desired',  color='m', where='post')
    ax1.legend(loc='upper left')
    ax1.grid('on')
    ax1.set(ylabel='Temperature [deg]')

    plt.show(block=False)
    plt.pause(0.2)
    """

##########################


def LoadModel(Tag,dt):
    if os.path.exists('Phi'+Tag+'_'+str(dt)+'_Desired_to_Indoor.pkl') and WarmStart:
        print('Load Regression Results '+Tag)
        f = open('Phi'+Tag+'_'+str(dt)+'_Desired_to_Indoor.pkl',"rb")
        Phi = pickle.load(f)
        f.close()
    else:
        Phi = {'Future' : {},
               'Past'   : {}}

        for label in Vars['Future']:
            Phi['Future'][label] = np.zeros([MSsetup['Pred'],MSsetup['Depth']+MSsetup['Pred']])
        for label in Vars['Past']:
            Phi['Past'][label] = np.zeros([MSsetup['Pred'],MSsetup['Depth']+1])
        print('No initial Phi')
    
    return Phi

def RegDiagLossAndGradient(phi):
    loss = 0
    grad = np.zeros(phi.shape)
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if i > 0 and j > 0:
                loss      += 0.5*(phi[i,j] - phi[i-1,j-1])**2
                grad[i,j] += phi[i,j] - phi[i-1,j-1]
                
            if i < phi.shape[0]-1 and j < phi.shape[1]-1:
                loss      += 0.5*(phi[i,j] - phi[i+1,j+1])**2
                grad[i,j] += phi[i,j] - phi[i+1,j+1]
    return loss, grad

def RegCausality(phi,type,decay):
    loss = 0
    grad = np.zeros(phi.shape)
    for i in range(phi.shape[0]):
        if type == 'Future':
            Range = i + phi.shape[1] - phi.shape[0] + 1
            for j in range(0,Range):
                # zero for j = Range - 1, any i
                # increases with i, decreases with j
                loss      += 0.5 * (1 - np.exp( - decay * (Range - j - 1)  )) * phi[i,j]**2
                grad[i,j] +=       (1 - np.exp( - decay * (Range - j - 1)  )) * phi[i,j]
        else:
            Range = phi.shape[1]
            for j in range(0,Range):
                # zero for j = Range-1, i = 0
                # increases with i, decreases with j
                loss      += 0.5 * (1 - np.exp( - decay * (i + Range - j - 1)  )) * phi[i,j]**2
                grad[i,j] +=       (1 - np.exp( - decay * (i + Range - j - 1)  )) * phi[i,j]
                                
    return loss, grad

def RegTimeLossAndGradient(phi,type):
    """
    figID = plt.figure(1,figsize=(5,5))
    ax1 = figID.add_subplot(121)
    ax1.imshow(grad,cmap=cmap)
    ax1 = figID.add_subplot(122)
    ax1.imshow(phi,cmap=cmap)
    plt.show(block=False)
    """

    loss = 0
    grad = np.zeros(phi.shape)

    for i in range(phi.shape[0]):
        if type == 'Future':
            Range = i + phi.shape[1] - phi.shape[0] + 1
        else:
            Range = phi.shape[1]
        for j in range(0,Range):
            if j > 0:
                loss      += 0.5*(phi[i,j] - phi[i,j-1])**2
                grad[i,j] += phi[i,j] - phi[i,j-1]
                
            if j < Range-1:
                loss      += 0.5*(phi[i,j] - phi[i,j+1])**2
                grad[i,j] += phi[i,j] - phi[i,j+1]
                
    return loss, grad

def PlotPhi(Phi, timegrid, Data, LossLog = [], Error = [], Tag = [], Label = []):

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.colors as colors
    
    cmap = 'seismic'

    vmax = 0
    for type in Vars:
        for label in Vars[type]:
            vmax = np.max([np.max(np.max(np.abs(Phi[type][label]),axis=0)),vmax])
            
    NumVars = len(Vars['Future']) + len(Vars['Past'])
    
    plt.close('all')
    figID = plt.figure(1,figsize=(10,5))
    k = 1
    for label in Vars['Future']:
        ax1 = figID.add_subplot(2,int(np.ceil(NumVars/2)),k)
        pcm = ax1.imshow(Phi['Future'][label],cmap=cmap,norm=colors.CenteredNorm(vcenter=0, halfrange=vmax))
        ax1.set_title(label)
        figID.colorbar(pcm, ax=ax1, extend='max')
        k += 1

    #figID = plt.figure(2,figsize=(10,6))

    for label in Vars['Past']:
        ax1 = figID.add_subplot(2,int(np.ceil(NumVars/2)),k)
        pcm = ax1.imshow(Phi['Past'][label],cmap=cmap,norm=colors.CenteredNorm(vcenter=0, halfrange=vmax))
        ax1.set_title(label)
        figID.colorbar(pcm, ax=ax1, extend='max')
        k += 1
            
    if LossLog:
        figID = plt.figure(3,figsize=(6,6))
        ax1 = figID.add_subplot(111)
        ax1.semilogy(range(len(LossLog)),LossLog/LossLog[0])
        ax1.grid('on')
    
    
    figID = plt.figure(4,figsize=(6,4))
    ax1 = figID.add_subplot(111)

    for type in Vars.keys():
        for label in Vars[type]:
            ax1.step(Data['Time'], Data[label],  color=Colors[label], label=label)
    
    ax1.set_title('Last data point at '+str(Data['Time'][-1]))
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='x', labelrotation=30)

    if not(isinstance(Error, list)):
        figID = plt.figure(5,figsize=(10,5))
        ax1  = figID.add_subplot(111)
        Std = np.max(np.std(Error,axis=1))
        Bins = np.linspace(-4*Std ,4*Std ,int(Error.shape[0]/15))
        [x,y] = np.meshgrid(timegrid,.5*Bins[1:] + .5*Bins[:-1])

        z = []
        for err in range(0,MSsetup['Pred']):
            a,_ = np.histogram(Error[:,err],bins=Bins)
            z.append(a/np.max(a))

        z = np.array(z).T
        plt.pcolormesh(x,y,z, cmap='Blues',shading='auto',vmin=z.min(), vmax=z.max())#, vmin=l_c, 'bone_r',shading='auto'
    
        mean = np.mean(Error.T,axis=1)
        plt.plot(timegrid , mean + np.std(Error.T,axis=1),color = 'r')
        plt.plot(timegrid , mean - np.std(Error.T,axis=1),color = 'r')
        plt.plot(timegrid , mean ,color = 'r')
        ax1.grid('on')
        if Tag:
            ax1.set_title(Tag+' : '+Label)

    plt.show(block=False)
    plt.pause(0.2)

timezone = pytz.timezone('Europe/Oslo')
start_indoor_sensor = datetime(2026,5,19,9,30).astimezone(timezone)

TrainingTime = Data['Time']

"""
# If we want to restrict the training to an early phase of the data
TrainingTime = []
for time in Data['Time']:
    if time < start_indoor_sensor :
        TrainingTime.append(time)
"""

print('Data window : '+str( np.round((Data['Time'][-1]-Data['Time'][0]).total_seconds()/3600/24))+' days')



# Show variables involved in the regression
Colors = { 'Indoor' : 'g', 'Supply' : 'r', 'Return' : [.75,0,0], 'Outdoor' : 'b', 'SetTemp' : 'k', 'Desired' : 'm', 'Feature' : 'c'}

# Weights and constraints used in the regression
for Tag in ['L2','L1Above1','L1Below1']: #

    if WarmStart and not(ResetL1):
        Phi = LoadModel(Tag, dt)
    else:
        Phi = LoadModel('L2', dt)

    if Tag == 'L1Below1':
        weight = {'plus'      : 0.99,   # Penalizes prediction > target
                  'minus'     : 0.01 ,  # Penalizes prediction < target
                  'regdiag'   : 0,     # Coherence on diagonals
                  'regtime'   : 0,     # Coherence in time
                  'regcausal' : {'weight' :  0,
                                 'decay'  : .01},    # Causality (vanishing infulence)
                  'L2vsL1'    : 0.0}    # 1 --> pure L2, 0 --> pure L1
        alpha    = 1e-6



    if Tag == 'L1Above1':
        weight = {'plus'      : 0.01,   # Penalizes prediction > target
                  'minus'     : 0.99,   # Penalizes prediction < target
                  'regdiag'   : 0,     # Coherence on diagonals
                  'regtime'   : 0,     # Coherence in time
                  'regcausal' : {'weight' : .0,
                                 'decay'  : .01},    # Causality (vanishing infulence)
                  'L2vsL1'    : 0.0}    # 1 --> pure L2, 0 --> pure L1
        alpha    = 1e-6


        
    if Tag == 'L2':
        weight = {'plus'      : 0   , # Penalizes prediction > target
                  'minus'     : 0   , # Penalizes prediction < target
                  'regdiag'   : 0,    # Coherence on diagonals
                  'regtime'   : 0,    # Coherence in time
                  'regcausal' : {'weight' : .0,
                                 'decay'  : .01},    # Causality (vanishing infulence)
                  'L2vsL1'    : 1}    # 1 --> pure L2, 0 --> pure L1
        alpha    = 1e-3
    


    CausalMask = np.ones([MSsetup['Pred'],MSsetup['Depth']+MSsetup['Pred']])
    for i in range(0,CausalMask.shape[0]):
        for j in range(0,CausalMask.shape[1]):
            if j > i + MSsetup['Depth']-1:
                CausalMask[i,j] = 0



    timegrid = [ i*dt/60 for i in range(MSsetup['Pred']) ]

    dLossQR  = {'Future' : {},
                'Past'   : {}}
    dLoss2   = {'Future' : {},
                'Past'   : {}}
    dLossReg = {'Future' : {},
                'Past'   : {}}

    dLossTot = {'Future' : {},
                'Past'   : {}}
                
    ### Run the iterations
    LossLog  = []
    LossSave = []

    
    print('Start Gradient Descent on '+Tag)
    print('Discount min : '+str(Discount ** ((Now-Data['Time'][0]).total_seconds()/3600/24)))
    iter = True
    niter = 0
    
    # Prepare homotopy for QR:
    wplus    = 0.
    wminus   = 0.
    homotopy = False
    if not(Tag == 'L2') and (WarmStart == False or ResetL1 == True):
        wplus  = 0.5
        wminus = 0.5
        if weight[  'plus'] > .5:
            wplus    = 0.8
            wminus   = 0.2
            homotopy = True
        if weight[ 'minus'] > .5:
            wplus  = 0.2
            wminus = 0.8
            homotopy = True
            
    # If QR is warm started, then no homotopy
    if WarmStart == True and ResetL1 == False:
        wplus    = weight[ 'plus']
        wminus   = weight['minus']
        homotopy = False
        
    while iter:

        N       = 0
        LossQR  = 0
        Loss2   = 0
        LossReg = 0

        for type in dLossQR.keys():
            for label in Vars[type]:
                dLossQR[type][label] = 0
                dLoss2[ type][label] = 0
                dLossReg[ type][label] = 0

        Error          = []
        Discount_scale = 0
        for k, time in enumerate(TrainingTime):
            DT_day = (Now-time).total_seconds()/3600/24
            
            if k >= MSsetup['Depth'] and k < len(Data['Time'])-MSsetup['Pred']:
                N += 1
                target = np.array([Data['Indoor'][ k+1:k+MSsetup['Pred']+1 ]]).T - Data['Indoor'][ k ]
                
                pred = 0
                for label in Vars['Future']:
                    u = np.array([Data[label][ k-MSsetup['Depth']:k+MSsetup['Pred'] ]]).T
                    pred += Phi['Future'][label].dot(u)

                for label in Vars['Past']:
                    u = np.array([Data[label][ k-MSsetup['Depth']:k+1 ]]).T
                    pred += Phi['Past'][label].dot(u)
                    
                
                error = pred - target  # > 0 => pred overestimates target / < 0 => pred underestimates target
                
                Error.append(error)
                
                ## Loss QR
                if weight['L2vsL1'] < 1:
                    signs   = wplus * (error > 0) - wminus * (error < 0)
                    LossQR += (Discount ** DT_day) * signs.T.dot(error)[0,0]

                ## Loss L2
                if weight['L2vsL1'] > 0:
                    Loss2  += (Discount ** DT_day) * 0.5*error.T.dot(error)[0,0]

                ## Gradients Future
                for label in Vars['Future']:
                    u = np.array([Data[label][ k-MSsetup['Depth']:k+MSsetup['Pred'] ]]).T
                    if weight['L2vsL1'] < 1:
                        dLossQR['Future'][label] += (Discount ** DT_day) * signs.dot(u.T)
                    if weight['L2vsL1'] > 0:
                        dLoss2[ 'Future'][label] += (Discount ** DT_day) * error.dot(u.T)
                    
                ## Gradients Past
                for label in Vars['Past']:
                    u = np.array([Data[label][ k-MSsetup['Depth']:k+1 ]]).T
                    if weight['L2vsL1'] < 1:
                        dLossQR['Past'][label] += (Discount ** DT_day) * signs.dot(u.T)
                    if weight['L2vsL1'] > 0:
                        dLoss2[ 'Past'][label] += (Discount ** DT_day) * error.dot(u.T)
                
                Discount_scale += Discount ** DT_day
                
        Error  = np.array(Error).squeeze()
            
        # Regularization
        for type in Vars.keys():
            for label in Vars[type]:
                if weight['regdiag'] > 0:
                    LossR, GradReg        = RegDiagLossAndGradient(Phi[type][label])
                    dLossReg[type][label] = dLossReg[type][label] + weight['regdiag'] * GradReg
                    LossReg              +=                         weight['regdiag'] * LossReg

                if weight['regtime'] > 0:
                    LossR, GradReg        = RegTimeLossAndGradient(Phi[type][label],type)
                    dLossReg[type][label] = dLossReg[type][label] + weight['regtime'] * GradReg
                    LossReg              +=                         weight['regtime'] * LossReg

                if weight['regcausal']['weight'] > 0:
                    LossR, GradReg        = RegCausality(Phi[type][label],type,weight['regcausal']['decay'])
                    dLossReg[type][label] = dLossReg[type][label] + weight['regcausal']['weight'] * GradReg
                    LossReg              +=                         weight['regcausal']['weight'] * LossReg

        # Total loss + scaling
        Loss  = LossQR + weight['L2vsL1'] * Loss2
        Loss *= 1 / Discount_scale / MSsetup['Pred']
        Loss += LossReg
        
        # Total gradient + scaling
        for type in Vars.keys():
            for label in Vars[type]:
                dLossTot[type][label] = (1-weight['L2vsL1']) * dLossQR[type][label] + weight['L2vsL1'] * dLoss2[type][label]
                dLossTot[type][label] = dLossTot[type][label] / Discount_scale / MSsetup['Pred']
                dLossTot[type][label] = dLossTot[type][label] + dLossReg[type][label]
        

        # Log loss
        if LossLog:
            if Loss > LossLog[-1]:
                alpha *= 0.99
                
        LossLog.append(Loss)
            
        # Take step
        for type in Vars.keys():
            for label in Vars[type]:
                Phi[type][label] -= alpha * dLossTot[type][label]

        # Project on causality
        for label in Vars['Future']:
            Phi['Future'][label] = Phi['Future'][label] * CausalMask
                
        ## Some verbose, intermediate save and plotting
        niter += 1
        if np.round(niter/100)-niter/100 == 0 or niter == 1:
            LossSave.append(Loss)
            save = True
            if LossSave:
                if not(LossSave[-1] == np.min(LossSave)):
                    save   = False
                if len(LossSave) > 1:
                    if LossSave[-1] > LossSave[-2]:
                        print('Adjust step size')
                        alpha *= 0.9
                    if 1 - LossSave[-1]/LossSave[-2] < ExitThreshold * (1 + 9 * homotopy):
                        if wplus == weight['plus'] and wminus == weight['minus']:
                            iter = False
                        else:
                            # Update QR weights
                            wplus  = 0.5 * wplus  + 0.5 * weight[ 'plus']
                            wminus = 0.5 * wminus + 0.5 * weight['minus']
                            
                            # Project if small and stop homotopy
                            if np.abs(wplus - weight[ 'plus']) + np.abs(wminus - weight['minus']) < 2e-3:
                                wplus  = weight[ 'plus']
                                wminus = weight['minus']
                                homotopy = False
                                
            print('Iter : '+str(niter)+' | Loss : '+str(Loss)+' | Alpha : '+str(np.log10(alpha))+' | L2 : '+str(weight['L2vsL1'])+' | w+ : '+str(wplus)+' | w- : '+str(wminus))

        # Plot every 10'000 iterates
        if Manual and np.round(niter/10000)-niter/10000 == 0:
            PlotPhi(Phi,timegrid, Data, LossLog, Error, Tag, Label = 'Desired --> Indoor')
            
            

    LenError = Error.shape[0]*Error.shape[1]
    print('------------------------------------------------------------')
    print('Error above : '+str(np.round(1000*np.sum(Error > 0)/LenError)/10)+'% | Error below : '+str(np.round(1000*np.sum(Error < 0)/LenError)/10)+'%')
    print('------------------------------------------------------------')

    Stats = {'Time'  : timegrid,
             'Error' : Error}
    
    Phi['Stats'] = Stats
    

    Phi['Vars'] = Vars
    

    f = open('Phi'+Tag+'_'+str(dt)+'_Desired_to_Indoor.pkl',"wb")
    pickle.dump(Phi,f, protocol=2)
    f.close()

    
    if Manual:
        PlotPhi(Phi,timegrid, Data, LossLog, Error, Tag, Label = 'Desired --> Indoor')


TrainTime = (datetime.now().astimezone(timez)-Now).total_seconds()/60
print('Training time : '+str(int(np.round(TrainTime)))+ ' min')
    
MSsetup[  'dt'] = dt
MSsetup['Vars'] = Vars

            
         
f = open('RegressionSetup.pkl',"wb")
pickle.dump(MSsetup,f, protocol=2)
f.close()
