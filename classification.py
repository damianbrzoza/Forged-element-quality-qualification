
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

df = pd.read_csv('annotations.csv', index_col=0)
save_configuration=[147, 11, 3, 3.15230084773765, 0.0608307017621155, 45, 3]

# Function for checking results of parameters optimalization
def check_results(dataset, congfiguration_list, plots=False):
    list_of_lines = []
    list_of_black = []
    images = [cv2.imread(name) for name in dataset['img']]

    for image in images:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if congfiguration_list[6] == 0:
            edges = cv2.Canny(img, congfiguration_list[0], congfiguration_list[1], apertureSize=congfiguration_list[2])
        elif congfiguration_list[6] == 1:
            edges = cv2.Sobel(img, cv2.CV_8UC1, 1, 0, ksize=congfiguration_list[2])
        else:
            edges = cv2.Canny(img, congfiguration_list[0], congfiguration_list[1], apertureSize=congfiguration_list[2])
            edges = cv2.Sobel(edges, cv2.CV_8UC1, 1, 0, ksize=congfiguration_list[2])

        list_of_black.append(edges.sum() / 255)
        lines = cv2.HoughLines(edges, congfiguration_list[3], congfiguration_list[4], congfiguration_list[5])
        # Printing lines on image
        if lines is not None:
            if plots:
                for line in lines:
                    for rho, theta in line:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * -b)
                        y1 = int(y0 + 1000 * a)
                        x2 = int(x0 - 1000 * -b)
                        y2 = int(y0 - 1000 * a)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                res = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
                cv2.imshow('obr1', res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            list_of_lines.append(len(lines))
        else:
            list_of_lines.append(0)
    dataset['lines'] = list_of_lines
    dataset['black_color'] = list_of_black

    dataset['bad_or_not'] = 1
    dataset['bad_or_not'][dataset['lines'] < 1250] = 0
#    dataset['bad_or_not'][dataset['lines'] < 1000] = 1
    list_of_acc = []
    for index,row in dataset.iterrows():
        if row['bad_or_not'] == 0 and 5 < row['rating'] <= 10:
            list_of_acc.append(1)
        elif row['bad_or_not'] == 2 and row['rating'] > 10:
            list_of_acc.append(1)
        elif row['bad_or_not'] == 1 and row['rating'] <= 5:
            list_of_acc.append(1)
        else:
            list_of_acc.append(0)
    dataset['acc'] = list_of_acc
    accuracy = sum(dataset['acc'])/len(dataset.index)
    print(accuracy)

    dataset = dataset.sort_values(by=['rating'])
    dataset.plot(x='rating', y='lines', kind='scatter',c='bad_or_not',colormap='jet')
    plt.show()
    dataset.plot(x='rating', y='black_color', kind='scatter')
    plt.show()

    dt = dataset.groupby('rating', as_index=False)[['lines', 'black_color']].mean()
    print(dataset)
    dt.plot(x='rating', y='lines', kind='scatter')
    #print(np.corrcoef(dt['rating'], dt['lines']))
    plt.show()


# Function for counting correlation of given instance of configuration
def get_correlation(configuration_list, dataset):
    list_of_lines = []
    list_of_black = []
    images = [cv2.imread(name) for name in dataset['img']]
    for image in images:

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if configuration_list[6] == 0:
            edges = cv2.Canny(img, configuration_list[0], configuration_list[1], apertureSize=configuration_list[2])
        elif configuration_list[6] == 1:
            edges = cv2.Sobel(img, cv2.CV_8UC1, 1, 0, ksize=configuration_list[2])
        else:
            edges = cv2.Canny(img, configuration_list[0], configuration_list[1], apertureSize=configuration_list[2])
            edges = cv2.Sobel(edges, cv2.CV_8UC1, 1, 0, ksize=configuration_list[2])

        list_of_black.append(edges.sum() / 255)
        lines = cv2.HoughLines(edges, configuration_list[3], configuration_list[4], configuration_list[5])
        if lines is not None:
            list_of_lines.append(len(lines))
        else:
            list_of_lines.append(0)

    dataset['lines'] = list_of_lines
    dataset['black_color'] = list_of_black

    dataset = dataset.sort_values(by=['rating'])

    dt = dataset.groupby('rating', as_index=False)[['lines', 'black_color']].mean()
    return np.corrcoef(dt['rating'], dt['lines'])[0][1]


# Function for make random configuration
def new_configuration(dataset):
    configuration = list()
    configuration.append(int(random.random() * 255))
    configuration.append(int(random.random() * 255))
    configuration.append(random.randrange(3, 7 + 1, 2))
    configuration.append((random.random() * 50)/10)
    configuration.append((random.random() * 10) * np.pi/180)
    configuration.append(int(random.random() * 255))
    configuration.append(random.randint(0, 3))
    return get_score(configuration, dataset), configuration


# Function to get_score
def get_score(configuration, dataset):
    return get_correlation(configuration, dataset)


# Random search algorithm to find best configuration
def random_search(dataset,iterations):
    best_score, best_configuration = new_configuration(dataset)
    print(best_configuration)
    y = [best_score]

    for i in range(iterations):
        score, configuration = new_configuration(dataset)
        print(configuration)
        if score > best_score:
            best_score, best_configuration = score, configuration
        y.append(best_score)
    x_dataset = list(range(len(y)))
    plt.plot(x_dataset, y, label="random_search")
    plt.ylabel('score')
    plt.xlabel('configuration')
    return best_score, best_configuration


best, instance = random_search(df,iterations=100)
print("Random Search")
print("Best score: " + str(best))
print("Best configuration " + str(instance))
plt.show()

check_results(df, instance)
