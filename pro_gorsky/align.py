from numpy import array, dstack, roll, vsplit, sum, reshape, dstack
from skimage.transform import rescale
from skimage.measure import compare_mse
from time import clock

def correct(image):
    image = image[round(image.shape[0] * 0.05) 
        : round(image.shape[0] * 0.95), round(image.shape[1] * 0.05) : round(image.shape[1] * 0.95)]
    return image

def alignforrange(dist, base, first, second):
    sumF = 162162162.0
    sumS = 162162162.0
    minNumF = (-dist, -dist)
    minNumS = (-dist, -dist)
    for lx in range(-dist, dist + 1):
        for ly in range(-dist, dist + 1):
            fir = roll(first, lx, axis = 0)
            fir = roll(fir, ly, axis = 1)
            sec = roll(second, lx, axis = 0)
            sec = roll(sec, ly, axis = 1)
            alx = abs(lx)
            aly = abs(ly)
            fir = fir[: fir.shape[0] - alx, : fir.shape[1] - aly]
            sec = sec[: sec.shape[0] - alx, : sec.shape[1] - aly]
            # print(fir.shape, sec.shape)

            shape = min(fir.shape[0], base.shape[0]) , min(fir.shape[1], base.shape[1])
            bas = base[ : shape[0], : shape[1]]
            fir = fir[ : shape[0]]
            sF = compare_mse(bas, fir)

            shape = min(sec.shape[0], base.shape[0]) , min(sec.shape[1], base.shape[1])
            bas = base[ : shape[0], : shape[1]]
            sec = sec[ : shape[0]]
            sS = compare_mse(bas, sec)

            if (sumF > sF):
                minNumF = (lx, ly)
                sumF = sF
                #print("fir changed:", sumF, minNumF)
            if (sumS > sS):
                minNumS = (lx, ly)
                sumS = sS
    return minNumF, minNumS

def align(bgr_image, g_coord):
    # start = clock()
    b_row = b_col = r_row = r_col = 0
    high = round(bgr_image.shape[0] / 3)
    imBlue, imGreen, imRed = bgr_image[: high], bgr_image[high: 2 * high], bgr_image[2 * high:]
    # images = vsplit(bgr_image, 3)
    tripToGreen = high
    tripToRed = high
    imBlue = correct(imBlue)
    imGreen = correct(imGreen)
    imRed = correct(imRed)

    base = imGreen
    first = imBlue
    second = imRed

    size = base.shape[0] * base.shape[1]

    # dich starting
    if (base.shape[1] > 500):
        sumlxF = 0
        sumlyF = 0
        sumlxS = 0
        sumlyS = 0
        scale = 8
        while (scale != 0):
            smallBase = rescale(base, 1 / scale)
            smallFirst = rescale(first, 1 / scale)
            smallSecond = rescale(second, 1 / scale)
            minNumF, minNumS = alignforrange(scale, smallBase, smallFirst, smallSecond)

            lx, ly = minNumF
            lx *= scale
            ly *= scale
            sumlxF += lx
            sumlyF += ly

            fir = roll(first, lx, axis = 0)
            fir = roll(fir, ly, axis = 1)
            alx = abs(lx)
            aly = abs(ly)
            fir = fir[: fir.shape[0] - alx, : fir.shape[1] - aly]

            lx, ly = minNumS
            lx *= scale
            ly *= scale
            sumlxS += lx
            sumlyS += ly

            sec = roll(second, lx, axis = 0)
            sec = roll(sec, ly, axis = 1)
            alx = abs(lx)
            aly = abs(ly)
            sec = sec[: sec.shape[0] - alx, : sec.shape[1] - aly]

            shape = min(sec.shape[0], base.shape[0], fir.shape[0]) , min(sec.shape[1], base.shape[1], fir.shape[1])
            base = base[ : shape[0], : shape[1]]
            second = sec[ : shape[0], : shape[1]]
            first = fir[: shape[0], : shape[1]]
            scale //= 2
        res = dstack((second, base, first))
        minNumF, minNumS = (sumlxF, sumlyF), (sumlxS, sumlyS)
    else:
        # print("returns", 16, time)
        minNumF, minNumS = alignforrange(16, base, first, second)

        lxF, lyF = minNumF
        fir = roll(first, lxF, axis = 0)
        fir = roll(fir, lyF, axis = 1)
        alx = abs(lxF)
        aly = abs(lyF)
        fir = fir[: fir.shape[0] - alx, : fir.shape[1] - aly]

        lxS, lyS = minNumS
        lxS += 3
        sec = roll(second, lxS, axis = 0)
        sec = roll(sec, lyS, axis = 1)
        alx = abs(lxS)
        aly = abs(lyS)
        sec = sec[: sec.shape[0] - alx, : sec.shape[1] - aly]

        shape = min(sec.shape[0], base.shape[0], fir.shape[0]) , min(sec.shape[1], base.shape[1], fir.shape[1])
        bas = base[ : shape[0], : shape[1]]
        sec = sec[ : shape[0], : shape[1]]
        fir = fir[: shape[0], : shape[1]]
        res = dstack((sec, bas, fir))
        minNumF, minNumS = (lxF, lyF), (lxS, lyS)

# get coordinates
    b_row = g_coord[0] - minNumF[0] - tripToGreen
    b_col = g_coord[1] - minNumF[1]
    r_row = g_coord[0] - minNumS[0] + tripToRed
    r_col = g_coord[1] - minNumS[1]

# get resulting image
    # print(clock() - start)
    return res, (b_row, b_col), (r_row, r_col)