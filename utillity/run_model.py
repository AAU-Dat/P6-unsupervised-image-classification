import numpy


def train_model(model, train_loader, device, break_after_2=False):
    # train the model
    n_total_steps = len(train_loader)
    print(f"\r0 out of {n_total_steps}", end="")
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        model.fit(images)
        print(f"\r{i} out of {n_total_steps}", end="")
        if break_after_2 and i == 2:
            break
    print("\n")


def evaluate_model(model, test_loader):
    clusters = 10
    eval = numpy.zeros((clusters, 10))
    for i, (images, labels) in enumerate(test_loader):
        res = model.predict(images)
        label = labels.numpy()
        for i in range(len(res)):
            eval[res[i]][label[i]] += 1
        break

    #self label
    reps = []
    for i in range(clusters):
        reps.append(0)

    for i in range(clusters):
        for j in range(10):
            if eval[i][j] > eval[i][reps[i]]:
                reps[i] = j


    # find correct and wrong answers
    right = []
    for i in range(clusters):
        right.append(0)

    wrong = []
    for i in range(clusters):
        wrong.append(0)

    for i in range(clusters):
        for j in range(10):
            if reps[i] != j:
                wrong[i] += eval[i][j]
            else:
                right[i] += eval[i][j]


    # find accuracy
    accuarcy = []
    for i in range(clusters):
        accuarcy.append(0)

    wrongs = 0
    rights = 0
    for i in range(10):
        wrongs += wrong[i]
        rights += right[i]
        if right[i] != 0 and (right[i] + wrong[i]) != 0:
            accuarcy[i] = right[i]/(right[i] + wrong[i])

    print(eval)
    print(reps)
    print(accuarcy)
    print(rights)
    print(wrongs)
    if rights != 0 and (rights + wrongs) != 0:
        print(rights/(rights + wrongs))