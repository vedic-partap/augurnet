import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    x_t = [1,2,3,4]
    x_d = [2,3,4,5]
    y_t = [4,3,2,1]
    y_d = [3,2,1,0]
    z_t = [1,-1,2,-2]
    z_d = [-1,2,-2,3]    
    plot(x_t,y_t,z_t,x_d,y_d,z_d)