import numpy as np 
from augurnet.utils import read_file, compare_interval_count
from augurnet.plotter import plot_host_events
from os import path
import errno
import os
import logging
import torch
from torch.utils.data.dataset import Dataset


class TemporalData:
    def __init__(self, events_filename, times_filename):
        # check if path exist
        if not path.exists(events_filename):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), events_filename)
        if not path.exists(times_filename):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), times_filename)
        
        self.events = read_file(events_filename)
        self.times = read_file(times_filename)
        assert self.events.shape == self.times.shape, "Events and Times must be of same dimensions"
    def get_host_count(self):
        if self.events is None or self.times is None:
            return None
        else:
            return len(self.events)

    def plot_data(self):
        plot_host_events(self.events, self.times)

    def describe(self):
        pass


class NTPPData(Dataset):
    def __init__(self, temporal_data, args):
        self.args = args
        self.type = 'train'
        events = temporal_data.events
        times = temporal_data.times
        self.time_step = args['time_step']
        self.host_count = len(events)
        train_interval = int(args['int_count'] * (1 - args['test_size']))
        interval_size = (max([s[-1] for s in times]) + 1) / args['int_count']

        interval_count = np.zeros((self.host_count, args['int_count']),
                                  dtype=int)

        self.train_event, self.train_times = [
            [] for i in range(self.host_count)
        ], [[] for i in range(self.host_count)]
        self.test_event, self.test_times = [[] for i in range(self.host_count)
                                  ], [[] for i in range(self.host_count)]
        for i, host in enumerate(times):
            for j, time_stamp in enumerate(host):
                counter = int(time_stamp / interval_size)
                interval_count[i][counter] += 1
                if counter < train_interval:
                    self.train_times[i].append(times[i][j])
                    # print(type(events[i][j]))
                    self.train_event[i].append(1.0)
                else:
                    self.test_times[i].append(times[i][j])
                    self.test_event[i].append(1.0)
        print(len(self.train_event),len(self.train_event[0]))
        min_count = min([len(x) for x in self.train_event])
        assert min_count >= args[
            'time_step'] + 2, "Time Step should be less than {0}".format(
                min_count - 2)
        self.train_y = compare_interval_count(0, train_interval,
                                              self.host_count, interval_count)

        self.test_y = compare_interval_count(train_interval, args['int_count'],
                                             self.host_count, interval_count)
        self.train_times = [x[:self.time_step + 2] for x in self.train_times]
        self.train_event = [x[:self.time_step + 2] for x in self.train_event]
        

    def getObservation(self):
        return self.train_y, self.test_y

    def startDev(self):
        self.type = 'dev'

    def startTrain(self):
        self.type = 'train'
    
    def change_data(self,times):
        
        # print(self.host_count)
        for i in range(self.host_count):
            last_value = self.train_times[i][-1]
            self.train_times[i].append(last_value+times[i])
        # self.train_times = [x[1:] for x in self.train_times]
            
    def __getitem__(self, idx):
        if self.type == 'train':
            return np.array(self.train_event[idx][:self.time_step + 2]), np.array(
                    self.train_times[idx][-2 - self.time_step:])
        elif self.type =='dev':
            return np.array(self.test_event[idx][:self.time_step + 2]), np.array(
                    self.test_times[idx][:2 + self.time_step])
    def get_times(self):
        return self.train_times
    def __len__(self):
        return self.host_count


if __name__ == '__main__':
    events_filename = "augurnet/data/preprocess/events.txt"
    times_filename  = "augurnet/data/preprocess/times.txt"
    data = TemporalData(events_filename, times_filename)
    print("Host count",data.get_host_count())
    # print(data.events)
        
