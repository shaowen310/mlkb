def n_params(net):
    nb_params = 0
    for param in net.parameters():
        nb_params += param.numel()
    return nb_params


# Credit Xavier Bresson CE7454 Github utils.py
def display_num_param(net):
    nb_params = n_params(net)
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_params, nb_params / 1e6))
