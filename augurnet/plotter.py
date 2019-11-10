import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="darkgrid")
"""
Function to plot the loss and MAPE

=============
Input : list for losses and arguments

Ouptut :
=============

"""
def plot(train_loss_list, train_mape_t_list, train_mape_n_list, dev_loss_list, dev_mape_t_list, dev_mape_n_list):
    f1 = plt.figure(1)
    plt.plot(train_loss_list)
    plt.plot(dev_loss_list)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Dev'], loc='upper left')
    # plt.show()
    # plt.close(f1)

    f2 = plt.figure(2)
    plt.plot(train_mape_t_list)
    plt.plot(dev_mape_t_list)
    plt.title('MAPE_t')
    plt.ylabel('MAPE_t')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Dev'], loc='upper left')
    # plt.show()
    # plt.close(f2)

    f3 = plt.figure(3)
    plt.plot(train_mape_n_list)
    plt.plot(dev_mape_n_list)
    plt.title('MAPE_n')
    plt.ylabel('MAPE_n')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Dev'], loc='upper left')
    plt.show()
    plt.close(f1)
    plt.close(f2)
    plt.close(f3)

def plot_host_events(events, times):
    host_cnt = len(events)
    plt.figure()
    for i,(event,time) in enumerate(zip(events,times)):
        plt.subplot(host_cnt,1,i+1)
        sns.lineplot(time,event)
        plt.xlabel("Time")
        plt.ylabel("Length of Packets")
        plt.title(f"Time series data for Host {i}")
    plt.show()

def plot_predictions(data):
    temporal_prediction = np.transpose(data,axes=(1,0))
    plt.figure()
    host_cnt = len(temporal_prediction)
    time_steps = len(temporal_prediction[0])
    for i,time in enumerate(temporal_prediction):
        plt.subplot(host_cnt,1,i+1)
        sns.lineplot(range(time_steps), time,marker='o')
        plt.xlabel("Time stamp")
        plt.ylabel("Time of Events")
        plt.title(f"Time predictions for Host {i}")
    plt.show()

if __name__ == "__main__":
    x_t = [1,2,3,4]
    x_d = [2,3,4,5]
    y_t = [4,3,2,1]
    y_d = [3,2,1,0]
    z_t = [1,-1,2,-2]
    z_d = [-1,2,-2,3]    
    plot(x_t,y_t,z_t,x_d,y_d,z_d)