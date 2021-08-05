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

def tsplot(ax,x,mean,std,**kwargs):
    cis = (mean - std, mean + std)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2,**kwargs)
    ax.plot(x,mean,lw=2,**kwargs)
    #ax.margins(x=0)

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

def get_simple_loss_array(seq,loss,epoch_size,batch_size):
    losses = np.zeros(len(seq))
    #print(loss)
    epoch=0
    #print(len(seq))

    for i in np.arange(len(seq)):
        j = int(i/batch_size)
        losses[i] = loss[epoch][j%epoch_size]
        if (j!=0) & (i%(epoch_size*batch_size)==0):
                epoch+=1

    return losses, np.arange(len(seq))


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
                    plt.plot(losses[key]['timestep'],losses[key]['val'], '.',marker='^',markersize=2,label=key,color='red')
        elif key == 'D':
            plt.plot(losses[key]['timestep'],losses[key]['val'], '.',markersize=2,label=key,color='dodgerblue')
        else:
                plt.plot(losses[key]['timestep'],losses[key]['val'], '.',markersize=2,label=key,color='k')
    plt.legend(losses.keys())
    plt.ylabel('loss')
    plt.xlabel('batch number')
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'scatter_big',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 


def plot_average_loss(losses,save_path, name):
    #print(losses)
    losses = np.array(losses)
    print(np.shape(losses))
    print(np.shape(np.mean(losses,0)))
    #print(len(losses))
    plt.figure(figsize=(3,3))
    ax13=plt.subplot(111)
    tsplot(ax13,np.arange(np.shape(losses)[1]),np.mean(losses,0),np.std(losses,0), color = 'k')
    plt.ylabel('loss')
    plt.xticks(np.arange(0,np.shape(losses)[1],2000),np.arange(0,int(np.shape(losses)[1]/10),200))
    plt.xlabel('batch number')
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'lossovertime',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
    print('average loss')
    print(np.mean(losses,0))


def plot_EversusDloss(losses,seq,save_path, name):
    #print(losses)
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
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'EversusDzoom',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

def plot_E_asfctof_Epos(losses,seq, batch_size, save_path, name):
    
    plt.figure(figsize=(3,3))
    for key in losses:
        if key == 'E':
            plt.plot(np.array(losses[key]['timestep'])%batch_size,losses[key]['val'], '.k',markersize=7,label=key)    
    plt.ylabel('E loss')
    plt.xlabel('position of frame in batch')
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
    plt.ylabel('E loss')
    plt.xlabel('number of Es in batch')

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

def plot_E_asfctof_loss2(losses,loss,seq,epoch_size,batch_size,save_path,name):
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
    plt.savefig('%s%s%s.pdf'%(save_path,'E_asfctof_loss_',name))
    
def get_dotproduct(dot_foreach,seq,loss,epoch_size,batch_size, surp_epoch, plot, name, save_path):
   
    # dot_foreach is a dictionary which for each subepoch contains a matrix 
    # [batch_size,1,16,batch_size,1,16] array of all the z_hat z dot products
    # the first 3 dim are for the z_hat, the other 3 are for the zs
    
    
    dot = {'A':{'match':[],'negatives':[],'batch':[],'loss':[]},
          'B':{'match':[],'negatives':[],'batch':[],'loss':[]},
          'C':{'match':[],'negatives':[],'batch':[],'loss':[]},
          'D':{'match':[],'negatives':[],'batch':[],'loss':[]},
          'E':{'match':[],'negatives':[],'batch':[],'loss':[]}}


    img_size = dot_foreach[0][1].shape[2]
    epoch=0

    for i, sequence in enumerate(seq):
        j = int(i/batch_size)

        #the matches are where the index in the third dimension is equal to the index in the  sixth dimension k=k 
        matches = [dot_foreach[epoch][j%epoch_size][int(i%batch_size),0,k,int(i%batch_size),0,k].item() for k in range(img_size)]
        #the matches are where the index in the third dimension is unequal to the index in the  sixth dimension k!=l         
        negatives = [dot_foreach[epoch][j%epoch_size][int(i%batch_size),0,k,int(i%batch_size),0,l].item() for k in range(img_size) for l in range(img_size) if k!=l]

        dot[sequence[-1]]['match'].append(np.mean(np.array(matches)))
        dot[sequence[-1]]['negatives'].append(np.mean(np.array(negatives)))
        dot[sequence[-1]]['batch'].append(j)
        dot[sequence[-1]]['loss'].append(loss[epoch][j%epoch_size][i%batch_size])

        if (j!=0) & (i%(epoch_size*batch_size)==0):
                epoch+=1
    
        E_match = dot['E']
        D_match = dot['D']
        
        
    if plot == True:
        plt.figure(figsize=(3,3))
        plt.plot(D_match['batch'],D_match['match'],'.k',markersize=7)
    
        plt.plot(E_match['batch'],E_match['match'],'.r',markersize=7)
        plt.plot(D_match['batch'],D_match['loss'],'b',markersize=4)
        plt.plot(E_match['batch'],E_match['loss'],'g',markersize=4)
    
        plt.xlabel('batch number')    
        plt.ylabel('dot product of matches <z_hat_i,z_i>')
        plt.legend(['D','E'])  
        plt.xlim(100,200)
        plt.tight_layout()
        plt.savefig('%s%s%s.pdf'%(save_path,'E_versus_D_matches_andloss',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
    
        plt.figure(figsize=(3,3))
        plt.plot(D_match['batch'],D_match['negatives'],'.k',markersize=7)
        plt.plot(E_match['batch'],E_match['negatives'],'.r',markersize=7)
        plt.xlabel('batch number')    
        plt.ylabel('dot product of negatives <z_hat_i,z_j> i!=j')
        plt.legend(['D match','E match','D loss', 'E loss'])    
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
            # go over all batches in which there is an E
            batches[batch] = batch
            Didxs = np.nonzero((batch<=D_match['batch'])&(D_match['batch']<batch+interval))
            Dminidx = np.min(Didxs)
            Dmaxidx = np.max(Didxs)
            # take the mean over the next interval (=10) batches (moving average)
            smooth_D_matches[batch] = np.mean(D_match['match'][Dminidx:Dmaxidx])
            smooth_D_negatives[batch] = np.mean(D_match['negatives'][Dminidx:Dmaxidx])
            Eidxs = np.nonzero((batch<=E_match['batch'])&(E_match['batch']<batch+interval))
            if len(Eidxs[0])>0:
                Eminidx = np.min(Eidxs)
                Emaxidx = np.max(Eidxs)        
                # take the mean over the next interval (=10) batches (moving average)
                smooth_E_matches[batch] = np.mean(E_match['match'][Eminidx:Emaxidx])            
                smooth_E_negatives[batch] = np.mean(E_match['negatives'][Eminidx:Emaxidx]) 
                # calculate the difference between E and D for the matches and the negatives
                Diff[batch]= np.mean(E_match['match'][Eminidx:Emaxidx]) - np.mean(D_match['match'][Dminidx:Dmaxidx])
                Neg_Diff[batch]= np.mean(E_match['negatives'][Eminidx:Emaxidx]) - np.mean(D_match['negatives'][Dminidx:Dmaxidx])
    
        plt.figure(figsize=(3,3))
        plt.plot(batches[smooth_D_matches!=0],smooth_D_matches[smooth_D_matches!=0],'k',markersize=7)
        plt.plot(batches[smooth_E_matches!=0],smooth_E_matches[smooth_E_matches!=0],'r',markersize=7)
        plt.xlabel('batch number')    
        plt.ylabel('<z_hat_i,z_i> over %d batches'%interval)    
        plt.legend(['D','E'])
        plt.xlim(100,200)
        plt.tight_layout()
        plt.savefig('%s%s%s.pdf'%(save_path,'startinterval10line_smooth_E_versus_D_matches_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
    
        plt.figure(figsize=(3,3))
        plt.plot(batches[smooth_D_negatives!=0],smooth_D_negatives[smooth_D_negatives!=0],'k',markersize=7)
        plt.plot(batches[smooth_E_negatives!=0],smooth_E_negatives[smooth_E_negatives!=0],'r',markersize=7)
        plt.xlabel('batch number')    
        plt.ylabel('<z_hat_i,z_j> i!=j over %d batches'%interval)    
        plt.legend(['D','E'])
        plt.xlim(100,200)
        plt.tight_layout()
        plt.savefig('%s%s%s.pdf'%(save_path,'startinterval10line_smooth_E_versus_D_negatives_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
    
    
        plt.figure(figsize=(3,3))
        plt.plot(batches[Diff!=0],Diff[Diff!=0],'.')    
        plt.xlabel('batch from start of suprises')    
        plt.ylabel('E - D matches')    
        plt.tight_layout()
        plt.savefig('%s%s%s.pdf'%(save_path,'start10Diff_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
    
        plt.figure(figsize=(3,3))
        plt.plot(batches[Neg_Diff!=0],Neg_Diff[Neg_Diff!=0],'.')    
        plt.xlabel('batch from start of suprises')    
        plt.ylabel('E - D negatives')    

        plt.tight_layout()
        plt.savefig('%s%s%s.pdf'%(save_path,'start10Neg_Diff_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

    return dot


def plot_average(dot_list,epoch_size,num_epochs,name,save_path):
    
    all_E_matches = np.zeros((len(dot_list),epoch_size*num_epochs))
    all_D_matches = np.zeros((len(dot_list),epoch_size*num_epochs))
    all_batches = np.zeros((len(dot_list),epoch_size*num_epochs))
    all_nonsmooth_E_matches = np.zeros((len(dot_list),epoch_size*num_epochs))
    all_nonsmooth_D_matches = np.zeros((len(dot_list),epoch_size*num_epochs))

    for i, item in enumerate(dot_list):
        E_match = item['E']
        D_match = item['D']
            
        interval = 10
        smooth_E_matches = np.zeros((np.max(E_match['batch'])+1))
        smooth_D_matches = np.zeros((np.max(E_match['batch'])+1))
        smooth_E_negatives = np.zeros((np.max(E_match['batch'])+1))
        smooth_D_negatives = np.zeros((np.max(E_match['batch'])+1))
        Diff = np.zeros((np.max(E_match['batch'])+1))
        nonsmooth_E_matches = np.zeros((np.max(E_match['batch'])+1))
        nonsmooth_D_matches = np.zeros((np.max(E_match['batch'])+1))

        Neg_Diff = np.zeros((np.max(E_match['batch'])+1))
        batches = np.zeros((np.max(E_match['batch'])+1))
        Ebatches = np.zeros((np.max(E_match['batch'])+1))
        Dbatches = np.zeros((np.max(E_match['batch'])+1))
        Sbatches = np.zeros((np.max(E_match['batch'])+1))
        for batch in np.unique(E_match['batch']):
            batches[batch] = batch
            Dbatchidx = np.nonzero((batch==D_match['batch']))
            Ebatchidx = np.nonzero((batch==E_match['batch']))
            if len(Dbatchidx[0])>0:
                Dbatches[batch] = batch
                Dbatchmin = np.min(Dbatchidx)
                Dbatchmax = np.max(Dbatchidx)
                nonsmooth_D_matches[batch] = np.mean(D_match['match'][Dbatchmin:Dbatchmax+1])
            if len(Ebatchidx[0])>0:
                Ebatches[batch] = batch
                Ebatchmin = np.min(Ebatchidx)
                Ebatchmax = np.max(Ebatchidx)
                nonsmooth_E_matches[batch] = np.nanmean(E_match['match'][Ebatchmin:Ebatchmax+1])
                #print(E_match['match'][Ebatchmin:Ebatchmax])
                #print(nonsmooth_E_matches)
                Eidxs = np.nonzero((batch<=E_match['batch'])&(E_match['batch']<batch+interval))
                Didxs = np.nonzero((batch<=D_match['batch'])&(D_match['batch']<batch+interval))
            if len(Eidxs[0]>0):
                #print('Didxs')
                #print(Didxs)
                if len(Didxs[0]>0):
                    
                    Sbatches[batch] = batch
                    #print('if fulfilled')
                    Dminidx = np.min(Didxs)
                    Dmaxidx = np.max(Didxs)
                    smooth_D_matches[batch] = np.mean(D_match['match'][Dminidx:Dmaxidx+1])
                    smooth_D_negatives[batch] = np.mean(D_match['negatives'][Dminidx:Dmaxidx+1])
                    #print(Eidxs)
                    Eminidx = np.min(Eidxs)
                    Emaxidx = np.max(Eidxs)        
                    smooth_E_matches[batch] = np.mean(E_match['match'][Eminidx:Emaxidx+1])            
                    smooth_E_negatives[batch] = np.mean(E_match['negatives'][Eminidx:Emaxidx+1])            
                    Diff[batch]= np.mean(E_match['match'][Eminidx:Emaxidx+1]) - np.mean(D_match['match'][Dminidx:Dmaxidx+1])
                    Neg_Diff[batch]= np.mean(E_match['negatives'][Eminidx:Emaxidx+1]) - np.mean(D_match['negatives'][Dminidx:Dmaxidx+1])

        all_E_matches[i,:len(smooth_E_matches[smooth_E_matches!=0])] = smooth_E_matches[smooth_E_matches!=0]
        #print(all_E_matches)
        all_D_matches[i,:len(smooth_D_matches[smooth_D_matches!=0])] = smooth_D_matches[smooth_D_matches!=0]
        all_batches[i,:len(smooth_E_matches[smooth_E_matches!=0])] = Sbatches[smooth_E_matches!=0]
        all_nonsmooth_E_matches[i,:len(nonsmooth_E_matches[nonsmooth_E_matches!=0])] = nonsmooth_E_matches[nonsmooth_E_matches!=0]
        all_nonsmooth_D_matches[i,:len(nonsmooth_D_matches[nonsmooth_E_matches!=0])] = nonsmooth_D_matches[nonsmooth_E_matches!=0]
        

    #print(np.unique(E_match['batch']))
    #print('%%%%%%%%%%%%%5')
    #print(np.shape(all_E_matches))
    #print(all_E_matches)
    #print(all_D_matches)
    #print(all_batches[0,:])
    #print(np.shape(all_batches[0,:]))
    mean_E_matches = np.nanmean(all_E_matches,0)
    mean_D_matches = np.nanmean(all_D_matches,0)
    mean_nonsmooth_E_matches = np.nanmean(all_nonsmooth_E_matches,0)
    mean_nonsmooth_D_matches = np.nanmean(all_nonsmooth_D_matches,0)
    std_E_matches = np.nanstd(all_E_matches,0)
    std_D_matches = np.nanstd(all_D_matches,0)
    
    mean_batches = np.mean(all_batches,0)

    diff1 = all_E_matches-all_D_matches
    diff = all_E_matches[:,mean_E_matches!=0]-all_D_matches[:,mean_E_matches!=0]
    mean_diff = np.nanmean(diff,0)
    std_diff = np.nanstd(diff,0)
    #print(mean_diff)
    mean_diff1 = np.nanmean(diff1,0)
    std_diff1 = np.nanstd(diff1,0)
    plt.figure(figsize=(3,3))
    #ax11 = plt.subplot(111)
    #for i in range(np.shape(all_D_matches)[0]):
    #    plt.plot(all_D_matches[i,:],color='dodgerblue')
    #    plt.plot(all_E_matches[i,:],color='red')
    #plt.xlabel('batch of batch number')    
    #plt.ylabel('<z_hat_i,z_i> over %d batches'%interval)    
    #plt.legend(['D','E'])
    #plt.xlim(0,100)
    #plt.tight_layout()
    #plt.savefig('%s%s%s.pdf'%(save_path,'all_E_versus_D_matches_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

    
    #print('mean matches')
    #print(mean_batches)
    #print(np.shape(mean_E_matches))
    #print(np.shape(mean_D_matches))
    #print(np.shape(mean_E_matches[mean_E_matches!=0]))
    #print(np.shape(std_E_matches[mean_E_matches!=0]))
    #print(np.shape(mean_D_matches[mean_D_matches!=0]))
    #print(np.shape(std_D_matches[mean_D_matches!=0]))
    #print('all_batches')

    #print(np.shape(all_batches[0,:][mean_D_matches!=0]))
    #print(all_batches[0,:][mean_D_matches!=0])
    #print(all_batches[0,:][mean_E_matches!=0])

    #print(mean_E_matches)
    #print(mean_D_matches[mean_D_matches!=0])
    #print(mean_E_matches[mean_E_matches!=0])
    #print(np.shape(std_D_matches))

    D_x_values = all_batches[0,:][mean_D_matches!=0]
    E_x_values = all_batches[0,:][mean_E_matches!=0]
    D_mean_values = mean_D_matches[mean_D_matches!=0]
    E_mean_values = mean_E_matches[mean_E_matches!=0]
    D_std_values = std_D_matches[mean_D_matches!=0]
    E_std_values = std_E_matches[mean_E_matches!=0]

    plt.figure(figsize=(3,3))
    ax14 = plt.subplot(111)
    tsplot(ax14,D_x_values[D_x_values!=0],D_mean_values[D_x_values!=0],D_std_values[D_x_values!=0],color='dodgerblue')
    tsplot(ax14,E_x_values[E_x_values!=0],E_mean_values[E_x_values!=0],E_std_values[D_x_values!=0],color='red')
    #tsplot(ax14,all_batches[0,:][mean_E_matches!=0],mean_E_matches[mean_E_matches!=0],std_E_matches[mean_E_matches!=0],color='red')
    plt.xlabel('batch number')    
    plt.ylabel('<z_hat_i,z_i> over %d batches'%interval)    
    plt.legend(['D','E'])
    plt.xlim(np.min(E_x_values[E_x_values!=0]),200)
    #plt.xlim(100,200)
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'average_E_versus_D_matches_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

   # plt.figure(figsize=(3,3)
   # plt.plot(all_batches[0,:],mean_nonsmooth_D_matches,color='dodgerblue',markersize=7)
    
   # plt.plot(all_batches[0,:],mean_nonsmooth_E_matches,color='red',markersize=7)
   # plt.xlabel('batch of batch number')    
   # plt.ylabel('<z_hat_i,z_i>')    
   # plt.legend(['D','E'])
   # plt.xlim(100,200)
   # plt.tight_layout()
   # plt.savefig('%s%s%s.pdf'%(save_path,'average_E_versus_D_nonsmooth_matches_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

    print(np.shape(mean_diff))
    print(mean_diff)
    plt.figure(figsize=(3,3))
    plt.plot(all_batches[0,:],mean_E_matches-mean_D_matches,'k',markersize=7)
    plt.xlabel('batch of batch number')    
    plt.ylabel('E minus D matches over %d batches'%interval)    
    plt.legend(['D','E'])
    plt.xlim(100,200)
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'average_E_minus_D_matches_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
    print(np.shape(mean_diff)[0])
    
    plt.figure(figsize=(3,3))
    ax3 = plt.subplot(111)
    tsplot(ax3,E_x_values[E_x_values!=0],mean_diff[E_x_values!=0],std_diff[E_x_values!=0],color='k')
    plt.xlabel('batch number')    
    plt.ylabel('E minus D matches over %d batches'%interval)    
    #plt.legend(['D','E'])
    plt.xlim(np.min(E_x_values[E_x_values!=0]),200)
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'average_diff_E_minus_D_matches_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 

    #plt.figure(figsize=(3,3))
    #ax12 = plt.subplot(111)
    #tsplot(ax12,all_batches[0,:],mean_diff1,std_diff1,color='k')
    #plt.xlabel('batch of batch number')    
    #plt.ylabel('E minus D matches over %d batches'%interval)    
    #plt.legend(['D','E'])
    #plt.xlim(0,100)
    #plt.tight_layout()
    #plt.savefig('%s%s%s.pdf'%(save_path,'average_diff1_E_minus_D_matches_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 




    Diff = []
    for i in range(int(len(D_match['match'])/interval)):
        if D_match['batch'][i] > 5*20:
            Diff.append(np.mean(E_match['match'][i:i+interval]) - np.mean(D_match['match'][i:i+interval])
    plt.figure(figsize=(3,3))
    plt.plot(Diff,'.k',markersize=7)
    plt.xlabel('batch number')    
    plt.ylabel('E-D matches')    
    plt.tight_layout()
    plt.savefig('%s%s%s.pdf'%(save_path,'E_minus_D_matches_',name))#, bbox_extra_artists=(lgd,), bbox_inches='tight') 
#       
#                 

def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--direc', default='.')
    args = parser.parse_args()

    name = 'pretrained_noblanks_noroll'
    name = 'pretrained_noblanks'
    #name = 'pretrained_noblanks_numseq2'
    #name = 'pretrained_noblanks_numseq4_Elast'
    #name = 'pretrained_noblanks_numseq4_Elast_earlysurprise'
    name = 'pretrained_noblanks_numseq4_Elast'
    name = 'pretrained_noblanks_numseq4_Elast_bothED_batch10'

    #name = 'pretrained_noblanks_noroll_numseq4_Elast'
    print('start')
    save_path = args.direc + '/' + name + '/'
    SE=5
    SEED=5

    np.set_printoptions(threshold=sys.maxsize)
    with open(r'%sloss_%d_%d.yaml'%(save_path,SE,SEED)) as file:
        loss = yaml.load(file, Loader=yaml.Loader)
    #
    with open(r'%sseq_%d_%d.yaml'%(save_path,SE,SEED)) as file:
        seq = yaml.load(file, Loader=yaml.Loader)
    #print('loaded seq')
    epoch_size = 20
    batch_size = 10
    num_epochs = 25
    surp_epoch = 5
    #plot_noblanks_noroll(loss,seq,batch_size,save_path,name+'%d%d'%(SE,SEED))
    if len(loss) != len(seq):
        loss_dict = get_losses2(seq,loss,epoch_size,batch_size)
    else:
        loss_dict = get_losses(seq,loss,epoch_size)
    #plot_E_asfctof_Epos(loss_dict,seq, batch_size, save_path, name+'%d%d'%(SE,SEED))
    #Ecount = plot_loss_asfctof_numberofEinbatch(loss_dict,seq, epoch_size, num_epochs, save_path, name+'%d%d'%(SE,SEED))
    plot_noblanks(loss_dict,seq,save_path,name+'%d%d'%(SE,SEED))
    #plot_E_asfctof_loss2(loss_dict,loss,seq,epoch_size,batch_size,save_path,name+'%d%d'%(SE,SEED))


    #with open(r'%sloss_foreach_%d_%d.yaml'%(save_path,SE,SEED)) as file:
    #    loss_foreach = yaml.load(file, Loader=yaml.Loader)
    #loss_dict = get_losses3(seq,loss_foreach,epoch_size,batch_size)
    #plot_noblanks(loss_dict,seq, save_path, name,name+'%d%d'%(SE,SEED),Ecount=None)
    #plot_EversusDloss(loss_dict,seq,save_path, name+'%d%d'%(SE,SEED))
        
    #with open(r'%sdot_foreach_%d_%d.yaml'%(save_path,SE,SEED)) as file:
    #    dot_foreach = yaml.load(file, Loader=yaml.Loader)

    #with open(r'%starget_foreach_%d_%d.yaml'%(save_path,SE,SEED)) as file:
    #    target_foreach = yaml.load(file, Loader=yaml.Loader)

    #print(dot_foreach[0][1].shape) 
    #print(target_foreach[0][1].shape) 

    #get_dotproduct(dot_foreach,seq,loss_foreach,epoch_size,batch_size, surp_epoch,True,name+'%d%d'%(SE,SEED),save_path)
    #print('here')
    dot_list = []
    loss_list = []
    for SEED in np.arange(2,31):
        with open(r'%sseq_%d_%d.yaml'%(save_path,SE,SEED)) as file:
            seq = yaml.load(file, Loader=yaml.Loader)

        with open(r'%sloss_%d_%d.yaml'%(save_path,SE,SEED)) as file:
            loss = yaml.load(file, Loader=yaml.Loader)
        loss_array, len_seq = get_simple_loss_array(seq,loss,epoch_size,batch_size)
        loss_list.append(loss_array)

        #print(np.shape(loss_array))
        #with open(r'%sseq_%d_%d.yaml'%(save_path,SE,SEED)) as file:
        #    seq = yaml.load(file, Loader=yaml.Loader)
        with open(r'%sdot_foreach_%d_%d.yaml'%(save_path,SE,SEED)) as file:
            dot_foreach = yaml.load(file, Loader=yaml.Loader)
        #print('loaded dotforeach')
        with open(r'%sloss_foreach_%d_%d.yaml'%(save_path,SE,SEED)) as file:
            loss_foreach = yaml.load(file, Loader=yaml.Loader)
        #print('loaded lossforeach')
        dot_list.append(get_dotproduct(dot_foreach,seq,loss_foreach,epoch_size,batch_size, surp_epoch, False, name+'%d%d'%(SE,SEED), save_path))
        print('appended')
    plot_average(dot_list,epoch_size,num_epochs,name,save_path)
    plot_average_loss(loss_list,save_path,name+'%d%d'%(SE,SEED))


if __name__ == '__main__':
    main()

