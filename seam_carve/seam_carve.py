from numpy import zeros, argmin, append, delete, insert, transpose, array, column_stack, row_stack
from math import sqrt

def gradient(matrix):
    resx = zeros(matrix.shape[0:2])
    resy = zeros(matrix.shape[0:2])
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if (x == 0):
                resx[0, y] = matrix[1, y] - matrix[0, y]
            elif (x == matrix.shape[0] - 1):
                resx[x, y] = matrix[x, y] - matrix[x - 1, y]
            else:
                resx[x, y] = matrix[x + 1, y] - matrix[x - 1, y]

            if (y == 0):
                resy[x, 0] = matrix[x, 1] - matrix[x, 0]
            elif (y == matrix.shape[1] - 1):
                resy[x, y] = matrix[x, y] - matrix[x, y - 1]
            else:
                resy[x, y] = matrix[x, y + 1] - matrix[x, y - 1]

    return (resx ** 2 + resy ** 2) ** (1 / 2)

def seam_carve(img, type, mask = None):
    orientation, mode = type.split()

    redIm = img[:, :, 0]
    greenIm = img[:, :, 1]
    blueIm = img[:, :, 2]
    bright = 0.299 * redIm + 0.587 * greenIm + 0.114 * blueIm
    brightness = gradient(bright)

    if mask is None:
        resized_mask = None
    else:
        # here should be anything with mask
        shape = img.shape[0] * img.shape[1]
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if (mask[x, y] > 0):
                    brightness[x, y] += shape * 256
                elif (mask[x, y] < 0):
                    brightness[x, y] -= shape * 256

    form = [img.shape[0], img.shape[1], 2]
    searchForMin = zeros(form)
    shape = searchForMin.shape
    null = [0, 0]
    carve_mask = zeros(img.shape[0:2])
    resized_img = zeros(null[0:2])
    resized_mask = zeros(null[0:2])
    if (orientation == 'horizontal'):
        if (mode == 'shrink'):
            for x in range(brightness.shape[0]):
                for y in range(brightness.shape[1]):
                    if (x == 0):
                        searchForMin[x, y, 0] = brightness[x, y]
                        searchForMin[x, y, 1] = -1
                    if (y == 0):
                        if (searchForMin[x - 1, y, 0] <= searchForMin[x - 1, y + 1, 0]):
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y, 0]
                            searchForMin[x, y, 1] = y
                        else:
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y + 1, 0]
                            searchForMin[x, y, 1] = y + 1
                    elif (y == brightness.shape[1] - 1):
                        if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x - 1, y, 0]):
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y - 1, 0]
                            searchForMin[x, y, 1] = y - 1
                        else:
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y, 0]
                            searchForMin[x, y, 1] = y
                    else:
                        if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x - 1, y, 0]):
                            if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x - 1, y + 1, 0]):
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y - 1, 0]
                                searchForMin[x, y, 1] = y - 1
                            else:
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y + 1, 0]
                                searchForMin[x, y, 1] = y + 1
                        else:
                            if (searchForMin[x - 1, y, 0] <= searchForMin[x - 1, y + 1, 0]):
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y, 0]
                                searchForMin[x, y, 1] = y
                            else:
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y + 1, 0]
                                searchForMin[x, y, 1] = y + 1

            iter = argmin(searchForMin[shape[0] - 1, :, 0])
            resized_img = delete(img[shape[0] - 1], iter)
            if mask is not None:
                resized_mask = delete(mask[shape[0] - 1], iter)
            carve_mask[shape[0] - 1, iter] = 1
            for x in range(shape[0] - 1, 0, -1):
                iter = int(searchForMin[x, iter, 1])
                # print(iter)
            # resized_img = append(delete(img[0], iter), resized_img, axis = 0)
                resized_img = row_stack((resized_img, delete(img[x - 1], iter)))
                carve_mask[x - 1, iter] = 1
                if mask is not None:
                    resized_mask = row_stack((resized_mask, delete(mask[x - 1], iter)))
                else:
                    resized_mask = None
        else:
            for x in range(brightness.shape[0]):
                for y in range(brightness.shape[1]):
                    if (x == 0):
                        searchForMin[x, y, 0] = brightness[x, y]
                        searchForMin[x, y, 1] = -1
                    if (y == 0):
                        if (searchForMin[x - 1, y, 0] <= searchForMin[x - 1, y + 1, 0]):
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y, 0]
                            searchForMin[x, y, 1] = y
                        else:
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y + 1, 0]
                            searchForMin[x, y, 1] = y + 1
                    elif (y == brightness.shape[1] - 1):
                        if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x - 1, y, 0]):
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y - 1, 0]
                            searchForMin[x, y, 1] = y - 1
                        else:
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y, 0]
                            searchForMin[x, y, 1] = y
                    else:
                        if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x - 1, y, 0]):
                            if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x - 1, y + 1, 0]):
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y - 1, 0]
                                searchForMin[x, y, 1] = y - 1
                            else:
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y + 1, 0]
                                searchForMin[x, y, 1] = y + 1
                        else:
                            if (searchForMin[x - 1, y, 0] <= searchForMin[x - 1, y + 1, 0]):
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y, 0]
                                searchForMin[x, y, 1] = y
                            else:
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y + 1, 0]
                                searchForMin[x, y, 1] = y + 1
            iter = argmin(searchForMin[shape[0] - 1, :, 0])
            resized_img = img[shape[0] - 1]
            right = img[shape[0] - 1, iter] + img[shape[0] - 1, iter + 1]
            resized_img = insert(resized_img, iter, right / 2)
            if mask is not None:
                resized_mask = mask[shape[0] - 1]
                resized_mask = insert(resized_mask, iter, 1)
            carve_mask[shape[0] - 1, iter] = 1
            # carve_mask[shape[0] - 1, iter + 1] = 1
            for x in range(shape[0] - 1, 0, -1):
                iter = int(searchForMin[x, iter, 1])
            # resized_img = append(delete(img[0], iter), resized_img, axis = 0)
                if (iter + 1 < img.shape[1]):
                    medium = img[x - 1, iter] + img[x - 1, iter + 1]
                    resized_img = row_stack((resized_img, insert(img[x - 1], iter, medium / 2)))
                    carve_mask[x - 1, iter] = 1
                    # carve_mask[x - 1, iter + 1] = 1
                    if mask is not None:
                        resized_mask = row_stack((resized_mask, insert(mask[x - 1], iter, 1)))
                    else:
                        resized_mask = None
                else:
                    medium = img[x - 1, iter] + img[x - 1, iter - 1]
                    resized_img = row_stack((resized_img, insert(img[x - 1], iter - 1, medium / 2)))
                    carve_mask[x - 1, iter] = 1
                    # carve_mask[x - 1, iter - 1] = 1
                    if mask is not None:
                        resized_mask = row_stack((resized_mask, insert(mask[x - 1], iter - 1, 1)))
                    else:
                        resized_mask = None
    else:
        if (mode == 'shrink'):
            for y in range(brightness.shape[1]):
                for x in range(brightness.shape[0]):
                    if (y == 0):
                        searchForMin[x, y, 0] = brightness[x, y]
                        searchForMin[x, y, 1] = -1
                    if (x == 0):
                        if (searchForMin[x, y - 1, 0] <= searchForMin[x + 1, y - 1, 0]):
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x, y - 1, 0]
                            searchForMin[x, y, 1] = x
                        else:
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x + 1, y - 1, 0]
                            searchForMin[x, y, 1] = x + 1
                    elif (x == brightness.shape[0] - 1):
                        if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x, y - 1, 0]):
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y - 1, 0]
                            searchForMin[x, y, 1] = x - 1
                        else:
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x, y - 1, 0]
                            searchForMin[x, y, 1] = x
                    else:
                        if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x, y - 1, 0]):
                            if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x + 1, y - 1, 0]):
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y - 1, 0]
                                searchForMin[x, y, 1] = x - 1
                            else:
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x + 1, y - 1, 0]
                                searchForMin[x, y, 1] = x + 1
                        else:
                            if (searchForMin[x, y - 1, 0] <= searchForMin[x + 1, y - 1, 0]):
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x, y - 1, 0]
                                searchForMin[x, y, 1] = x
                            else:
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x + 1, y - 1, 0]
                                searchForMin[x, y, 1] = x + 1
            y = shape[1] - 1
            x = 0
            minVal = searchForMin[x, y, 0]
            iter = 0
            for x in range(1, shape[0]):
                if (searchForMin[x, y, 0] < minVal):
                    minVal = searchForMin[x, y, 0]
                    iter = x
            resized_img = array(delete(img[:,shape[1] - 1], iter))
            # transpose(resized_img)
            if mask is not None:
                resized_mask = delete(mask[:,shape[1] - 1], iter)
            carve_mask[iter, shape[1] - 1] = 1
            for x in range(shape[1] - 1, 0, -1):
                iter = int(searchForMin[iter, x, 1])
                resized_img = column_stack((resized_img, delete(img[:, x - 1], iter)))
                carve_mask[iter, x - 1] = 1
                if mask is not None:
                    resized_mask = column_stack((resized_mask, delete(mask[:, x - 1], iter)))
                else:
                    resized_mask = None
        else:
            for y in range(brightness.shape[1]):
                for x in range(brightness.shape[0]):
                    if (y == 0):
                        searchForMin[x, y, 0] = brightness[x, y]
                        searchForMin[x, y, 1] = -1
                    if (x == 0):
                        if (searchForMin[x, y - 1, 0] <= searchForMin[x + 1, y - 1, 0]):
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x, y - 1, 0]
                            searchForMin[x, y, 1] = x
                        else:
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x + 1, y - 1, 0]
                            searchForMin[x, y, 1] = x + 1
                    elif (x == brightness.shape[0] - 1):
                        if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x, y - 1, 0]):
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y - 1, 0]
                            searchForMin[x, y, 1] = x - 1
                        else:
                            searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x, y - 1, 0]
                            searchForMin[x, y, 1] = x
                    else:
                        if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x, y - 1, 0]):
                            if (searchForMin[x - 1, y - 1, 0] <= searchForMin[x + 1, y - 1, 0]):
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x - 1, y - 1, 0]
                                searchForMin[x, y, 1] = x - 1
                            else:
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x + 1, y - 1, 0]
                                searchForMin[x, y, 1] = x + 1
                        else:
                            if (searchForMin[x, y - 1, 0] <= searchForMin[x + 1, y - 1, 0]):
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x, y - 1, 0]
                                searchForMin[x, y, 1] = x
                            else:
                                searchForMin[x, y, 0] = brightness[x, y] + searchForMin[x + 1, y - 1, 0]
                                searchForMin[x, y, 1] = x + 1
            y = shape[1] - 1
            x = 0
            minVal = searchForMin[x, y, 0]
            iter = 0
            for x in range(1, shape[0]):
                if (searchForMin[x, y, 0] < minVal):
                    minVal = searchForMin[x, y, 0]
                    iter = x
            resized_img = img[:,shape[1] - 1]
            if (iter != shape[0] - 1):
                right = img[iter, shape[1] - 1] + img[iter + 1, shape[1] - 1]
                resized_img = insert(resized_img, iter + 1, right / 2)
                if mask is not None:
                    resized_mask = mask[:,shape[1] - 1]
                    resized_mask = insert(resized_mask, iter + 1, 1)
            else:
                right = img[iter, shape[1] - 1] + img[iter - 1, shape[1] - 1]
                resized_img = insert(resized_img, iter, right / 2)
                if mask is not None:
                    resized_mask = mask[:,shape[1] - 1]
                    resized_mask = insert(resized_mask, iter, 1)
            carve_mask[iter, shape[1] - 1] = 1
            # carve_mask[iter + 1, shape[1] - 1] = 1
            for x in range(shape[1] - 1, 0, -1):
                iter = int(searchForMin[iter, x, 1])
            # resized_img = append(delete(img[0], iter), resized_img, axis = 0)
                if (iter + 1 < img.shape[0]):
                    medium = img[iter, x - 1] + img[iter + 1, x - 1]
                    resized_img = column_stack((resized_img, insert(img[:, x - 1], iter, medium / 2)))
                    carve_mask[iter, x - 1] = 1
                    # carve_mask[iter + 1, x - 1] = 1
                    if mask is not None:
                        resized_mask = column_stack((resized_mask, insert(mask[:, x - 1], iter, 1)))
                    else:
                        resized_mask = None
                else:
                    medium = img[iter, x - 1] + img[iter - 1, x - 1]
                    resized_img = column_stack((resized_img, insert(img[:, x - 1], iter - 1, medium / 2)))
                    carve_mask[iter, x - 1] = 1
                    # carve_mask[iter - 1, x - 1] = 1
                    if mask is not None:
                        resized_mask = column_stack((resized_mask, insert(mask[:, x - 1], iter - 1, 1)))
                    else:
                        resized_mask = None
    return (resized_img, resized_mask, carve_mask)
    # retest