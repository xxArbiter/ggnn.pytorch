import torch
from torch.autograd import Variable

def test(dataloader, net, criterion, optimizer, opt):
    test_loss = 0
    correct = 0
    net.eval()
    for i, (adj_matrix, annotation, target) in enumerate(dataloader, 0):
        padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 2)
        if opt.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()

        init_input = Variable(init_input)
        adj_matrix = Variable(adj_matrix)
        annotation = Variable(annotation)
        target = Variable(target)

        output = net(init_input, annotation, adj_matrix)

        test_loss += criterion(output, target).data

    test_loss /= len(dataloader.dataset)
    print('Test set: MSE loss: {:.4f}'.format(test_loss*10000))
