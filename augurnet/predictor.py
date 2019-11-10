import argparse, time
import torch
import numpy as np
import random
from augurnet.data import NTPPData
from augurnet.utils import ensure_dir
from augurnet.data import TemporalData ######
from augurnet.model import NTPP
from augurnet.scorer import discriminatorLoss, calculateLoss
from augurnet.plotter import plot, plot_predictions

class Augur:
    args = {
        'save_dir':'augurnet/data/saved/',
        'window_size':64,
        'int_count':100,
        'test_size':0.2,
        'time_step':8,
        'batch_size':11,
        'element_size':2,
        'h':32,
        'nl':1,
        'seed':123456,
        # 'mode',
        'epochs':100,
        'workers':4,
        'learning_rate':0.01,
        'metric':'AUC',
        'is_cuda':True,
        'optim':'Adam',
    }
    def __init__(self, **kwrags):
        # init. the paramters and the torch environment
        self.update(kwrags)
        torch.set_default_tensor_type('torch.DoubleTensor')

        np.random.seed(Augur.args['seed'])
        torch.manual_seed(Augur.args['seed'])
        random.seed(Augur.args['seed'])
        self.device = torch.device('cuda') if torch.cuda.is_available() and Augur.args['is_cuda'] else torch.device('cpu')
    
    def update(self,args):
        for arg,value in args.items():
            if arg in Augur.args:
                Augur.args[arg] = value

    def print_args(self):
        print(Augur.args)
    
    def fit(self,temporal_data,verbose=0, save_file=None):
        #  Code for training the model
        #model file --
        self.data = NTPPData(temporal_data, Augur.args)
        ensure_dir(Augur.args['save_dir'])
        #Loading Data

        #  Get the obsevation
        train_y,test_y = self.data.getObservation()
        if verbose: print('NTPP model...')
        self.model = NTPP(Augur.args, output_layer_size=1)

        #  Data using dataloader
        train_loader = torch.utils.data.DataLoader(self.data,
                                batch_size=Augur.args['batch_size'],
                                shuffle=False,
                                num_workers=Augur.args['workers'],
                                pin_memory=Augur.args['is_cuda'] # CUDA only
                                )
        #  Depending on the argument decide the optimizer
        if Augur.args['optim'] == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=Augur.args['learning_rate'])
        elif Augur.args['optim'] == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=Augur.args['learning_rate'])
        elif Augur.args['optim'] == 'RMS':
            optimizer = torch.optim.RMSprop(
                self.model.parameters(), lr=Augur.args['learning_rate'])
        else:
            raise Exception('use Proper optimizer')
        # Data struture to store teh loss and MAPE values of Dev and train set
        train_loss_list,train_mape_t_list, train_mape_n_list = [],[],[]
        dev_loss_list,dev_mape_t_list, dev_mape_n_list = [],[],[]

        # For each epochs
        for epoch_cnt,epoch in enumerate(range(Augur.args['epochs'])):
            #  Set the dataloader to Train
            self.data.startTrain()
            for i, (batch_events,batch_times) in enumerate(train_loader):
                start_time = time.time()
                # batch preprocess
                time_step = Augur.args['time_step']

                # Some preprocessing
                batch_events_part1 = batch_events[:,:time_step]
                batch_times_diff = batch_times[:, 1:1+time_step] - batch_times[:,:time_step]
                batch_times_diff_next = batch_times[:, 2:2+time_step] - batch_times[:,1:1+time_step]
                batch_times_diff = batch_times_diff[:,:,None] # exapnd dim in axis 2
                batch_events_part1 = batch_events_part1[:,:,None] # expand Dim
                batch_input = torch.cat((batch_times_diff,batch_events_part1),2)

                #forward pass
                outputs = self.model(batch_input)
                # print("outputs",outputs.size())
                last_time_step_layer = outputs.clone()
                last_time_step_layer = (last_time_step_layer.detach().numpy()).transpose()[-1]
                # print("Last_time_step",[1/(x+1e-9) for x in last_time_step_layer])
                predicted = []
                # print("last_time_step_layer shape: ", len(last_time_step_layer))
                for host in range(last_time_step_layer.shape[0]):
                    for secon_host in range(host+1,last_time_step_layer.shape[0]):
                        predicted.append(np.greater(last_time_step_layer[host],last_time_step_layer[secon_host]))
                discriminator_loss = discriminatorLoss(train_y, predicted, Augur.args['metric'])
                loss, mape_t, mape_n = calculateLoss(discriminator_loss, outputs, batch_times_diff_next, time_step)

                # save the loss for the plotting
                train_loss_list.append(loss)
                train_mape_t_list.append(mape_t)
                train_mape_n_list.append(mape_n)

                #backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch_cnt%100==0:
                    if verbose:  print('[TRAIN] Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} MAPE_t: {:4f} MAPE_n: {:4f} Time : {}'
                        .format(epoch, Augur.args['epochs'], i+1, len(train_loader), loss.item(), mape_t.item(), mape_n.item(), time.time()-start_time))

            # Set the dataoader to dev set
            self.data.startDev()
            with torch.no_grad():
                for i, (batch_events,batch_times) in enumerate(train_loader):
                    start_time = time.time()
                    # batch preprocess
                    time_step = Augur.args['time_step']
                    batch_events_part1 = batch_events[:,:time_step]
                    batch_times_diff = batch_times[:, 1:1+time_step] - batch_times[:,:time_step]
                    batch_times_diff_next = batch_times[:, 2:2+time_step] - batch_times[:,1:1+time_step]
                    batch_times_diff = batch_times_diff[:,:,None] # exapnd dim in axis 2
                    batch_events_part1 = batch_events_part1[:,:,None] # expand Dim
                    batch_input = torch.cat((batch_times_diff,batch_events_part1),2)

                    #forward pass
                    outputs = self.model(batch_input)
                    last_time_step_layer = outputs.clone()
                    last_time_step_layer = (last_time_step_layer.detach().numpy()).transpose()[-1]
                    predicted = []
                    for host in range(last_time_step_layer.shape[0]):
                        for secon_host in range(host+1,last_time_step_layer.shape[0]):
                            predicted.append(np.greater(last_time_step_layer[host],last_time_step_layer[secon_host]))
                    discriminator_loss = discriminatorLoss(train_y, predicted, Augur.args['metric'])
                    loss, mape_t, mape_n = calculateLoss(discriminator_loss, outputs, batch_times_diff_next, time_step)
                    dev_loss_list.append(loss)
                    dev_mape_t_list.append(mape_t)
                    dev_mape_n_list.append(mape_n)
                    if epoch_cnt%100==0:
                        if verbose:  print('[VALIDATION] Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} MAPE_t: {:4f} MAPE_n: {:4f} Time : {}'
                            .format(epoch, Augur.args['epochs'], i+1, len(train_loader), loss.item(), mape_t.item(), mape_n.item(), time.time()-start_time))

        # Plot the loss and MAPE
        self.train_meta_data = [train_loss_list, train_mape_t_list, train_mape_n_list,
            dev_loss_list, dev_mape_t_list, dev_mape_n_list]
        if save_file is not None:
            torch.save(self.model, save_file)

    def plot_losses(self):
        plot(*self.train_meta_data)

    def predict(self, time_step=10, model_file=None, verbose=0):
        if model_file is not None:
            self.model = torch.load(model_file)
        
    #  Data using dataloader
        train_loader = torch.utils.data.DataLoader(self.data,
                                batch_size=Augur.args['batch_size'],
                                shuffle=False,
                                num_workers=Augur.args['workers'],
                                pin_memory=Augur.args['is_cuda'] # CUDA only
                                )
        
        self.data.startTrain()
        self.time_predictions = []
        for run in range(time_step):
            for i, (batch_events,batch_times) in enumerate(train_loader):
                start_time = time.time()
                # batch preprocess 
                time_step = Augur.args['time_step']

                # Some preprocessing
                batch_events_part1 = batch_events[:,:time_step]
                batch_times_diff = batch_times[:, 1:1+time_step] - batch_times[:,:time_step]
                batch_times_diff = batch_times_diff[:,:,None] # exapnd dim in axis 2
                batch_events_part1 = batch_events_part1[:,:,None] # expand Dim
                batch_input = torch.cat((batch_times_diff,batch_events_part1),2)

                #forward pass
                outputs = self.model(batch_input)
                # print("outputs",outputs.size())
                last_time_step_layer = outputs.clone()
                last_time_step_layer = (last_time_step_layer.detach().numpy()).transpose()[-1]
                times = [1/(x+1e-9) for x in last_time_step_layer]
                if verbose: print("Last_time_step",times)
                self.time_predictions.append(times)
                self.data.change_data(times)

    def plot_next_events(self,*args,**kwrags):
        plot_predictions(self.time_predictions)


if __name__ == "__main__":
    events_filename = "augur/data/preprocess/events.txt"
    times_filename  = "augur/data/preprocess/times.txt"
    data = TemporalData(events_filename, times_filename)
    print("Host count",data.get_host_count())
    data.plot_data()
    model = Augur()
    model.fit(data)
    model.plot_losses()
    model.predict(verbose=1)
    model.plot_next_events()
            