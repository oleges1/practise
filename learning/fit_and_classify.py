from numpy import ones, arctan2, zeros, append, empty, dot
from skimage.filters import sobel_h, sobel_v
from skimage.transform import resize
from math import pi
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

def way(grad):
    if (grad == pi):
        return 8
    return int(9 * grad / pi)

def extract_hog(img):
    image = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    image = resize(image, (64, 64), mode = 'reflect')
    sobelx = sobel_v(image)
    sobely = sobel_h(image)
    modul_grad = (sobelx ** 2 + sobely ** 2) ** (1 / 2)
    way_grad = abs(arctan2(sobely, sobelx))
    gistogram_places = zeros((8, 8, 9))
    for x in range(8):
        for y in range(8):
            for i in range(8):
                for j in range(8):
                    pixelx = 8 * x + i
                    pixely = 8 * y + j
                    gistogram_places[x, y, way(way_grad[pixelx, pixely])] += modul_grad[pixelx, pixely]
    eps = 0.0000000001
    for x in range(7):
        for y in range(7):
            v = gistogram_places[x, y]
            v = append(v, gistogram_places[x + 1, y])
            v = append(v, gistogram_places[x, y + 1])
            v = append(v, gistogram_places[x + 1, y + 1])
            if ((x == 0) and (y == 0)):
                res = v / ((dot(v, v) + eps) ** (1 / 2))
                created = 0
            else:
                res = append(res, v / ((dot(v, v) + eps) ** (1 / 2)))
    return res

def fit_and_classify(train_features, train_labels, test_features):
    """
    #- cross-validation -#
    score = 0.0
    for i in range(10):
        print("Spliting...")
        input_train, input_test, ans_train, ans_test = train_test_split(train_features, train_labels, train_size = 0.69)
        print("Training...")
        clf = LinearSVC().fit(input_train, ans_train)
        print("Trained, please wait for testing...")
        score_i = clf.score(input_train, ans_train)
        print("Score for autist test", i + 1, "-", score_i)
        res = clf.predict(input_test)
        print("Konushin style:")
        print('Accuracy: %.4f' % (sum(ans_test == res) / float(ans_test.shape[0])))
        score_i = clf.score(input_test, ans_test)
        print("Score for test", i + 1, "-", score_i)
        score += score_i
    score /= 10
    print("Overall score:", score)
    """
    clf = LinearSVC().fit(train_features, train_labels)
    return clf.predict(test_features)
