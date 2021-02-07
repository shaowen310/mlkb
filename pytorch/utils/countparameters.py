def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# Credit Xavier Bresson CE7454 Github utils.py
def display_num_param(net):
    nb_params = count_params(net)
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_params, nb_params / 1e6))
