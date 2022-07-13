import torch
import numpy as np
from tqdm.notebook import tqdm
import config
import os
import calendar
import evaluation as E
import time

def train_nn(nn, train_loader, valid_loader, optimizer, lossfunction, device='cpu'):
    # dir for save temporary files
    if not os.path.exists('./temp'):
        os.mkdir('./temp')
    
    # create an unique ID for saving temp file, avoiding file overwriting while multiple training
    training_ID = ts = int(calendar.timegm(time.gmtime()))
    print(f'The ID for this training is {training_ID}.')
    
    # initialize best valid loss for saving the best model
    best_valid_loss = 10 ** 10
    
    # arrays to save training process
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    # to count the epoch without any improvement, for early stop
    patience = 0
    
    # timer
    start_time = time.process_time()
    
    # training
    for epoch in range(10**10):
        
        
        # some temp variables to calculate the loss and acc from mini-batch to batch
        num_of_mini_batch  = []
        loss_of_mini_batch = []
        acc_of_mini_batch  = []
        for X_train, y_train in train_loader:
            # reshape 2D images into 1D, also transfer to device
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            # forward propagation
            prediction_train = nn(X_train)
            
            # calculate loss
            train_loss_mini_batch = lossfunction(prediction_train, y_train)

            # calculate predicted class of input data
            yhat_train = torch.argmax(prediction_train.data, 1)
            
            # calculate how many predictions are correct
            train_correct = torch.sum(yhat_train == y_train.data)
            
            # calculate accuracy of prediction
            train_acc_mini_batch = train_correct / y_train.numel()

            # update parameters in model
            optimizer.zero_grad()
            train_loss_mini_batch.backward()
            optimizer.step()

            # loss and acc from mini-batch
            num_of_mini_batch.append(X_train.shape[0])
            loss_of_mini_batch.append(train_loss_mini_batch.item())
            acc_of_mini_batch.append(train_acc_mini_batch.item())

        # convert and record loss/acc from mini-batch to batch
        train_loss = np.average(loss_of_mini_batch, weights=num_of_mini_batch)
        train_losses.append(train_loss)
        train_acc = np.average(acc_of_mini_batch, weights=num_of_mini_batch)
        train_accs.append(train_acc)

        # similar as training, calculate loss and accuracy on valid data
        num_of_mini_batch  = []
        loss_of_mini_batch = []
        acc_of_mini_batch  = []
        with torch.no_grad():
            for X_valid, y_valid in valid_loader:
                X_valid = X_valid.to(device)
                y_valid = y_valid.to(device)
                prediction_valid = nn(X_valid) 
                valid_loss_mini_batch = lossfunction(prediction_valid, y_valid).data
                yhat_valid = torch.argmax(prediction_valid.data, 1)
                valid_correct = torch.sum(yhat_valid == y_valid.data)
                valid_acc_mini_batch = valid_correct / y_valid.numel()

                num_of_mini_batch.append(X_valid.shape[0])
                loss_of_mini_batch.append(valid_loss_mini_batch.item())
                acc_of_mini_batch.append(valid_acc_mini_batch.item())

            valid_loss = np.average(loss_of_mini_batch, weights=num_of_mini_batch)
            valid_losses.append(valid_loss)
            valid_acc = np.average(acc_of_mini_batch, weights=num_of_mini_batch)
            valid_accs.append(valid_acc)

        # if valid loss in current epoch is better than previous one, save this model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(nn, f'./temp/NN_temp_{training_ID}')
            random_state = torch.random.get_rng_state()
            torch.save(random_state, f'./temp/NN_temp_random_state_{training_ID}')
            patience = 0
        # if not, that means in this epoch, the model is not improved.
        else:
            patience += 1
            if patience > config.patience:
                print('Early stop.')
                break

        # print information about current epoch
        if epoch % 100 == 0:
            # timer
            end_time = time.process_time()
            
            print(f'| epoch: {epoch:-4d} | train acc: {train_acc*100:-.2f} | loss: {train_loss:-.3e} | valid acc: {valid_acc*100:-.2f} | loss: {valid_loss:-.3e} | time: {end_time-start_time:-.2e} |')
            
            # timer
            start_time = time.process_time()
            
    print('Finished.')
    
    return torch.load(f'./temp/NN_temp_{training_ID}'), train_losses, valid_losses, train_accs, valid_accs

def train_spnn(spnn, X_trains, y_trains, X_valids, y_valids, optimizer, lossfunction, train_factor, valid_factor, acc_factor, alpha):
    # dir for save temporary files
    if not os.path.exists('./temp'):
        os.mkdir('./temp')
    
    # create an unique ID for saving temp file, avoiding file overwriting while multiple training
    training_ID = ts = int(calendar.timegm(time.gmtime()))
    print(f'The ID for this training is {training_ID}.')
    
    # initialize best valid loss for saving the best model
    best_valid_loss = 10 ** 10
    
    # arrays to save training process
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    # to count the epoch without any improvement, for early stop
    patience = 0
    
    # timer
    start_time = time.process_time()
        
    # training
    for epoch in range(10**10):

        # forward propagation
        prediction_trains = spnn(X_trains)

        # calculate loss
        train_loss = lossfunction(prediction_trains, y_trains,train_factor)\
                     + alpha*spnn.GetNorm(config.pnorm)
        # calculate accuracy of prediction
        train_acc = E.ACC(prediction_trains, y_trains, acc_factor)

        # update parameters in model
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # save train performance
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # similar as training, calculate loss and accuracy on valid data
        with torch.no_grad():
            prediction_valids = spnn(X_valids) 
            valid_loss = lossfunction(prediction_valids, y_valids,valid_factor).data\
                         + alpha*spnn.GetNorm(config.pnorm)
            valid_acc = E.ACC(prediction_valids, y_valids, acc_factor)

        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # if valid loss in current epoch is better than previous one, save this model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(spnn, f'./temp/SPNN_temp_{training_ID}')
            random_state = torch.random.get_rng_state()
            torch.save(random_state, f'./temp/SPNN_temp_random_state_{training_ID}')
            patience = 0
        # if not, that means in this epoch, the model is not improved.
        else:
            patience += 1
            if patience > config.patience:
                print('Early stop.')
                break
        
        # timer
        end_time = time.process_time()
        
        # print information about current epoch
        if epoch % 100 == 0:
            # timer
            end_time = time.process_time()
        
            print(f'| epoch: {epoch:-4d} | TA: {train_acc*100:-.2f} | TL: {train_loss:-.3e} | VA: {valid_acc*100:-.2f} | VL: {valid_loss:-.3e} | Norm: {spnn.GetNorm():-.2e} | time: {end_time-start_time:-.2e} |')
            
            # timer
            start_time = time.process_time()
            
    print('Finished.')
    
    return torch.load(f'./temp/SPNN_temp_{training_ID}'), train_losses, valid_losses, train_accs, valid_accs
