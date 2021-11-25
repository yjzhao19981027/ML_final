import torch.utils.data
import torch.backends.cudnn as cudnn
from dataUtils.getLoader import getLoader


def test(Net):
    cuda = True if torch.cuda.is_available() else False
    cudnn.benchmark = True if torch.cuda.is_available() else False

    dataloader = getLoader(train=False)

    """ test """

    net = Net
    net = net.eval()

    if cuda:
        net = net.cuda()

    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        img, label = data_iter.next()

        batch_size = len(label)

        if cuda:
            img = img.cuda()
            label = label.cuda()

        class_output = net(img)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
