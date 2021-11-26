def splittingTo4(train_dataset):
    patch1 = []
    patch2 = []
    patch3 = []
    patch4 = []
    # 从本地硬盘上读取一条数据 (包括1张图像及其对应的标签)
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        new_data1 = image[:, :20, :20], label
        patch1.append(new_data1)

        new_data2 = image[:, :20, 12:], label
        patch2.append(new_data2)

        new_data3 = image[:, 12:, :20], label
        patch3.append(new_data3)

        new_data4 = image[:, 12:, 12:], label
        patch4.append(new_data4)
    return patch1, patch2, patch3, patch4
