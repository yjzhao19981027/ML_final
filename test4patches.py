import torch.utils.data
import torch.backends.cudnn as cudnn
from dataUtils.getCifar10 import getCifar10
from torch.utils.data import DataLoader
from dataUtils.splitting import splittingTo4


def test4patches(Net):
    cuda = True if torch.cuda.is_available() else False
    cudnn.benchmark = True if torch.cuda.is_available() else False

    test_dataset = getCifar10(train=False)
    patch1, patch2, patch3, patch4 = splittingTo4(test_dataset)
    patch1Loader = DataLoader(dataset=patch1, batch_size=64)
    patch2Loader = DataLoader(dataset=patch2, batch_size=64)
    patch3Loader = DataLoader(dataset=patch3, batch_size=64)
    patch4Loader = DataLoader(dataset=patch4, batch_size=64)

    net = Net
    net = net.eval()

    if cuda:
        net = net.cuda()

    len_dataloader = len(patch1Loader)
    patch1_iter = iter(patch1Loader)
    patch2_iter = iter(patch2Loader)
    patch3_iter = iter(patch3Loader)
    patch4_iter = iter(patch4Loader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        img_patch1, label = patch1_iter.next()
        img_patch2, _ = patch2_iter.next()
        img_patch3, _ = patch3_iter.next()
        img_patch4, _ = patch4_iter.next()

        batch_size = len(label)

        if cuda:
            img_patch1 = img_patch1.cuda()
            img_patch2 = img_patch2.cuda()
            img_patch3 = img_patch3.cuda()
            img_patch4 = img_patch4.cuda()
            label = label.cuda()

        class_output = net(img_patch1, img_patch2, img_patch3, img_patch4)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
