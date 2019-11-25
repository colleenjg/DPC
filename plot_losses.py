#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:00:09 2019

@author: wilmeska
"""
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_losses(seq,loss,epoch_size):
    losses = {'A':{'val':[],'timestep':[]},
              'B':{'val':[],'timestep':[]},
              'C':{'val':[],'timestep':[]},
              'D':{'val':[],'timestep':[]},
              'E':{'val':[],'timestep':[]},}
    
    epoch=0
    for i, sequence in enumerate(seq):
        losses[sequence[-1]]['val'].append(loss[epoch][i%epoch_size])
        losses[sequence[-1]]['timestep'].append(i)
    
        if (i!=0) & (i%epoch_size==0):
                epoch+=1
    return losses

def get_losses2(seq,loss,epoch_size,batch_size):
    losses = {'A':{'val':[],'timestep':[]},
              'B':{'val':[],'timestep':[]},
              'C':{'val':[],'timestep':[]},
              'D':{'val':[],'timestep':[]},
              'E':{'val':[],'timestep':[]},}
    
    epoch=0
    for i, sequence in enumerate(seq):
        j = int(i/batch_size)
        losses[sequence[-1]]['val'].append(loss[epoch][j%epoch_size])
        losses[sequence[-1]]['timestep'].append(j)
        if (j!=0) & (i%(epoch_size*batch_size)==0):
                epoch+=1
    return losses


def get_losses3(seq,loss,epoch_size,batch_size):
    losses = {'A':{'val':[],'timestep':[]},
              'B':{'val':[],'timestep':[]},
              'C':{'val':[],'timestep':[]},
              'D':{'val':[],'timestep':[]},
              'E':{'val':[],'timestep':[]},}
    
    epoch=0
    for i, sequence in enumerate(seq):
        j = int(i/batch_size)
        losses[sequence[-1]]['val'].append(loss[epoch][j%epoch_size][i%batch_size])
        losses[sequence[-1]]['timestep'].append(i)
    
        if (j!=0) & (i%(epoch_size*batch_size)==0):
                epoch+=1
    return losses    

def plot_noblanks_noroll(loss,seq, epoch_size, save_path, name):

    regular = []
    surprise = []
    regulartimes = []
    surprisetimes = []
    
    epoch=0
    for i, sequence in enumerate(seq):
        if sequence == ['A','B','C','D']:
            regular.append(loss[epoch][i%epoch_size])
            regulartimes.append(i)
        elif sequence == ['A','B','C','E']:
            surprise.append(loss[epoch][i%epoch_size])
            surprisetimes.append(i)
        else:
            raise ValueError
        
        if (i!=0) & (i%epoch_size==0):
                epoch+=1
            
    plt.figure(figsize=(3,3))
    plt.plot(regular, c='k',label='regular')
    plt.plot(surprise, c='r',label='surprise')
    plt.savefig('%s%s.pdf'%(save_path,'compare'))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
    
    plt.figure(figsize=(3,3))
    plt.plot(regulartimes,regular, 'k.',label='ABCD')
    plt.plot(surprisetimes,surprise, 'r.',label='ABCE')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('frame number')
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'scatter',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

def plot_noblanks(losses,seq, save_path, name,Ecount=None):
    
    #plt.figure(figsize=(3,3))
    #plt.plot(regular, c='k')
    #plt.plot(surprise, c='r')
    #plt.savefig('%s%s.pdf'%(save_path,'compare'))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
    #print(losses)
    plt.figure(figsize=(3,3))
    for key in losses:
        if key == 'E':
            if Ecount is not None:
                for i, timestep in enumerate(losses[key]['timestep']):
                    if Ecount[timestep] == 1:
                        c = 'k'
                    elif Ecount[timestep] == 2:
                        c = 'g'
                    elif Ecount[timestep] == 3:
                        c = 'r'
                    else:
                        c = 'b'
                    plt.plot(timestep,losses[key]['val'][i], '.',color = c, markersize=7,label=key)    
            else:
                plt.plot(losses[key]['timestep'],losses[key]['val'], '.',markersize=7,label=key)    
        else:
            plt.plot(losses[key]['timestep'],losses[key]['val'], '.',markersize=2,label=key)
    plt.legend(losses.keys())
    plt.ylabel('loss')
    plt.xlabel('frame number')
    #plt.ylim(0,25)
    #plt.xlim(100,200)
    #plt.title(name)
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'scatter_big',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

def plot_EversusDloss(losses,seq,save_path, name):
    print(losses)
    plt.figure(figsize=(3,3))
    for key in losses:
        if key == 'E':
            plt.plot(losses[key]['timestep'],losses[key]['val'], '.k',markersize=5,label=key)    
        elif key == 'D':
            plt.plot(losses[key]['timestep'],losses[key]['val'], '.r',markersize=5,label=key)    
        else:
            plt.plot(losses[key]['timestep'],losses[key]['val'], '.',markersize=2,label=key)
    plt.legend(losses.keys())
    plt.ylabel('loss')
    plt.xlabel('frame number')
    #plt.ylim(0,25)
    #plt.xlim(1000,2000)
    #plt.title(name)
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'EversusDzoom',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

def plot_E_asfctof_Epos(losses,seq, batch_size, save_path, name):
    
    plt.figure(figsize=(3,3))
    for key in losses:
        if key == 'E':
            plt.plot(np.array(losses[key]['timestep'])%batch_size,losses[key]['val'], '.k',markersize=7,label=key)    
    #plt.legend(losses.keys())
    plt.ylabel('E loss')
    plt.xlabel('position of frame in batch')
    #plt.ylim(0,25)
    #plt.xlim(100,200)
    #plt.title(name)
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'E_asfctof_Epos_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

def plot_loss_asfctof_numberofEinbatch(losses,seq, epoch_size, num_epochs, save_path, name):
    Ecount = np.zeros((epoch_size*num_epochs))
    Loss = np.zeros((epoch_size*num_epochs))
    
    plt.figure(figsize=(3,3))
    for key in losses:
        if key == 'E':
            for i,timestep in enumerate(losses[key]['timestep']):
                Ecount[timestep] += 1 
                Loss[timestep] = losses[key]['val'][i]
    plt.plot(Ecount,Loss, '.k',markersize=7,label=key)    
    #plt.legend(losses.keys())
    plt.ylabel('E loss')
    plt.xlabel('number of Es in batch')
    #plt.ylim(0,25)
    #plt.xlim(100,200)
    #plt.title(name)
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'Loss_asfctof_Enumberinbatch_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
    return Ecount



    
def plot_E_asfctof_loss(losses,seq,save_path,name):
    lossi = []
    E_times = losses['E']['timestep']
    for E_time in E_times:
        for key, subkeys in losses.items():
            if key != 'E':
                for subkey, value in subkeys.items():
                    if subkey == 'timestep':
                        idx = np.nonzero(np.array(value)==E_time-1)
                        if len(idx[0])>0:
                            lossi.append(losses[key]['val'][idx[0][0]])

    plt.figure(figsize=(3,3))
    plt.plot(lossi,losses['E']['val'],'.k',markersize=7)
    plt.xlabel('loss before E appeared')    
    plt.ylabel('loss when E appeared')    
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'E_asfctof_loss_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

def plot_E_asfctof_loss2(losses,loss,seq,epoch_size,save_path,name):
    lossi = []
    E_times = losses['E']['timestep']
    for E_time in E_times:
        batch = int(E_time/batch_size)
        epoch = int(E_time/epoch_size)
        lossi.append(loss[epoch][(E_time%epoch_size)-1])

    plt.figure(figsize=(3,3))
    plt.plot(lossi,losses['E']['val'],'.k',markersize=7)
    plt.xlabel('loss before E appeared')    
    plt.ylabel('loss when E appeared')    
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'E_asfctof_loss_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
                 

def get_dotproduct(dot_foreach,seq,epoch_size,batch_size, surp_epoch):
   
    # dot_foreach is a dictionary which for each subepoch contains a matrix 
    # [batch_size,1,16,batch_size,1,16] array of all the z_hat z dot products
    # the first 3 dim are for the z_hat, the other 3 are for the zs
    
    
    dot = {'A':{'match':[],'negatives':[],'batch':[]},
          'B':{'match':[],'negatives':[],'batch':[]},
          'C':{'match':[],'negatives':[],'batch':[]},
          'D':{'match':[],'negatives':[],'batch':[]},
          'E':{'match':[],'negatives':[],'batch':[]},}


    img_size = dot_foreach[0][1].shape[2]
    epoch=0
    print('img_size')
    print(img_size)
    for i, sequence in enumerate(seq):
        j = int(i/batch_size)
        print('look')
        print((i%batch_size))
        
        matches = [dot_foreach[epoch][j%epoch_size][int(i%batch_size),0,k,int(i%batch_size),0,k].item() for k in range(img_size)]
        print('matches')
        print(len(matches))
        negatives = [dot_foreach[epoch][j%epoch_size][int(i%batch_size),0,k,int(i%batch_size),0,l].item() for k in range(img_size) for l in range(img_size) if k!=l]
        print('negatives')
        print(len(negatives))

        dot[sequence[-1]]['match'].append(np.mean(np.array(matches)))
        dot[sequence[-1]]['negatives'].append(np.mean(np.array(negatives)))
        dot[sequence[-1]]['batch'].append(j)
        #actual[sequence[-1]]['vector'].append(actual_foreach[epoch][j%epoch_size][int((i%batch_size)*img_size):int((i%batch_size)*img_size+img_size)])
        #print(actual_foreach[epoch][j%epoch_size][int((i%batch_size)*img_size):int((i%batch_size)*img_size+img_size)])
        #actual[sequence[-1]]['batch'].append(j)    
        if (j!=0) & (i%(epoch_size*batch_size)==0):
                epoch+=1
    
        E_match = dot['E']
        D_match = dot['D']
        
        
#    dot_product = np.zeros(epoch_size)
#    for b in range(epoch_size):
#        E_vector_pred = E_pred['vector'][E_pred['batch']==b]
#        E_vector_actual = E_actual['vector'][E_actual['batch']==b]
#        #E_vector_pred = E_vector_pred.view(batch_size,-1)       
#        #E_vector_actual = E_vector_actual.view(batch_size,-1)
#        D_vector_pred = E_pred['vector'][E_pred['batch']==b]
#        D_vector_actual = E_actual['vector'][E_actual['batch']==b]
#        #D_vector_pred = E_vector_pred.view(batch_size,-1)       
#        #D_vector_actual = E_vector_actual.view(batch_size,-1)
#        print('E_vector')
#        print(E_vector_pred.shape)
        
        
        
        #dot_product[b] = torch.dot(E_vector_pred, E_vector_actual) 
        
    
    plt.figure(figsize=(3,3))
    plt.plot(D_match['batch'],D_match['match'],'.k',markersize=7)

    plt.plot(E_match['batch'],E_match['match'],'.r',markersize=7)
    plt.xlabel('batch number')    
    plt.ylabel('dot product of matches <z_hat_i,z_i>')
    plt.legend(['D','E'])    
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'E_versus_D_matches_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

    plt.figure(figsize=(3,3))
    plt.plot(D_match['batch'],D_match['negatives'],'.k',markersize=7)
    plt.plot(E_match['batch'],E_match['negatives'],'.r',markersize=7)
    plt.xlabel('batch number')    
    plt.ylabel('dot product of negatives <z_hat_i,z_j> i!=j')
    plt.legend(['D','E'])    
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'E_versus_D_negatives_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

    interval = 10
    smooth_E_matches = np.zeros((np.max(E_match['batch'])+1))
    smooth_D_matches = np.zeros((np.max(E_match['batch'])+1))
    smooth_E_negatives = np.zeros((np.max(E_match['batch'])+1))
    smooth_D_negatives = np.zeros((np.max(E_match['batch'])+1))
    Diff = np.zeros((np.max(E_match['batch'])+1))
    Neg_Diff = np.zeros((np.max(E_match['batch'])+1))
    batches = np.zeros((np.max(E_match['batch'])+1))
    for batch in np.unique(E_match['batch']):
        batches[batch] = batch
        Didxs = np.nonzero((batch<=D_match['batch'])&(D_match['batch']<batch+interval))
        Dminidx = np.min(Didxs)
        Dmaxidx = np.max(Didxs)
        smooth_D_matches[batch] = np.mean(D_match['match'][Dminidx:Dmaxidx])
        smooth_D_negatives[batch] = np.mean(D_match['negatives'][Dminidx:Dmaxidx])
        Eidxs = np.nonzero((batch<=E_match['batch'])&(E_match['batch']<batch+interval))
        if len(Eidxs)>0:
            Eminidx = np.min(Eidxs)
            Emaxidx = np.max(Eidxs)        
            smooth_E_matches[batch] = np.mean(E_match['match'][Eminidx:Emaxidx])            
            smooth_E_negatives[batch] = np.mean(E_match['negatives'][Eminidx:Emaxidx])            
            Diff[batch]= np.mean(E_match['match'][Eminidx:Emaxidx]) - np.mean(D_match['match'][Dminidx:Dmaxidx])
            Neg_Diff[batch]= np.mean(E_match['negatives'][Eminidx:Emaxidx]) - np.mean(D_match['negatives'][Dminidx:Dmaxidx])

    plt.figure(figsize=(3,3))
    plt.plot(batches[smooth_D_matches!=0],smooth_D_matches[smooth_D_matches!=0],'k',markersize=7)
    plt.plot(batches[smooth_E_matches!=0],smooth_E_matches[smooth_E_matches!=0],'r',markersize=7)
    plt.xlabel('batch of batch number')    
    plt.ylabel('<z_hat_i,z_i> over %d batches'%interval)    
    plt.legend(['D','E'])
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'dots_smooth_E_versus_D_matches_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

    plt.figure(figsize=(3,3))
    plt.plot(batches[smooth_D_negatives!=0],smooth_D_negatives[smooth_D_negatives!=0],'k',markersize=7)
    plt.plot(batches[smooth_E_negatives!=0],smooth_E_negatives[smooth_E_negatives!=0],'r',markersize=7)
    plt.xlabel('batch of batch number')    
    plt.ylabel('<z_hat_i,z_j> i!=j over %d batches'%interval)    
    plt.legend(['D','E'])
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'dots_smooth_E_versus_D_negatives_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 


    plt.figure(figsize=(3,3))
    plt.plot(batches[Diff!=0],Diff[Diff!=0])    
    plt.xlabel('batch from start of suprises')    
    plt.ylabel('E - D matches')    
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'Diff_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

    plt.figure(figsize=(3,3))
    plt.plot(batches[Neg_Diff!=0],Neg_Diff[Neg_Diff!=0])    
    plt.xlabel('batch from start of suprises')    
    plt.ylabel('E - D negatives')    
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'Neg_Diff_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 


#    Diff = []
#    for i in range(int(len(D_match['match'])/interval)):
#        if D_match['batch'][i] > 5*20:
#            Diff.append(np.mean(E_match['match'][i:i+interval]) - np.mean(D_match['match'][i:i+interval])
#    plt.figure(figsize=(3,3))
#    plt.plot(Diff,'.k',markersize=7)
#    plt.xlabel('batch number')    
#    plt.ylabel('E-D matches')    
#    plt.tight_layout()
#    plt.savefig('%s%s%s.pdf'%(save_path,'E_minus_D_matches_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
#       
#                 

name = 'pretrained_noblanks_noroll'
name = 'pretrained_noblanks'
#name = 'pretrained_noblanks_numseq2'
#name = 'pretrained_noblanks_numseq4_Elast'
#name = 'pretrained_noblanks_numseq4_Elast_earlysurprise'
name = 'pretrained_noblanks_numseq4_Elast'
name = 'pretrained_noblanks_numseq4_Elast_bothED_batch10'

#name = 'pretrained_noblanks_noroll_numseq4_Elast'

save_path = '/network/tmp1/wilmeska/'+name+'/'
SE=5
SEED=2

with open(r'%sloss_%d_%d.yaml'%(save_path,SE,SEED)) as file:
    loss = yaml.load(file, Loader=yaml.Loader)

with open(r'%sseq_%d_%d.yaml'%(save_path,SE,SEED)) as file:
    seq = yaml.load(file, Loader=yaml.Loader)

epoch_size = 20
batch_size = 10
num_epochs = 25
surp_epoch = 5
#plot_noblanks_noroll(loss,seq,batch_size,save_path,name+'%d%d'%(SE,SEED))
if len(loss) != len(seq):
    loss_dict = get_losses2(seq,loss,epoch_size,batch_size)
else:
    loss_dict = get_losses(seq,loss,epoch_size)
plot_E_asfctof_Epos(loss_dict,seq, batch_size, save_path, name+'%d%d'%(SE,SEED))
Ecount = plot_loss_asfctof_numberofEinbatch(loss_dict,seq, epoch_size, num_epochs, save_path, name+'%d%d'%(SE,SEED))
plot_noblanks(loss_dict,seq,save_path,name+'%d%d'%(SE,SEED),Ecount)
plot_E_asfctof_loss2(loss_dict,loss,seq,epoch_size,save_path,name+'%d%d'%(SE,SEED))


with open(r'%sloss_foreach_%d_%d.yaml'%(save_path,SE,SEED)) as file:
    loss_foreach = yaml.load(file, Loader=yaml.Loader)
loss_dict = get_losses3(seq,loss_foreach,epoch_size,batch_size)
#plot_noblanks(loss_dict,seq, save_path, name,name+'%d%d'%(SE,SEED),Ecount=None)
plot_EversusDloss(loss_dict,seq,save_path, name+'%d%d'%(SE,SEED))
    
with open(r'%sdot_foreach_%d_%d.yaml'%(save_path,SE,SEED)) as file:
    dot_foreach = yaml.load(file, Loader=yaml.Loader)

with open(r'%starget_foreach_%d_%d.yaml'%(save_path,SE,SEED)) as file:
    target_foreach = yaml.load(file, Loader=yaml.Loader)

#print(dot_foreach[0][1].shape) 
#print(target_foreach[0][1].shape) 

get_dotproduct(dot_foreach,seq,epoch_size,batch_size, surp_epoch)


