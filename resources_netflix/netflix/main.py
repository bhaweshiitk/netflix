import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")
# TODO: Your code here
# for i in range(5):
# K=3
# i=0
# mixture, post = common.init(X, K, i)
# title= "EM Model with K=" + str(K) + " and " + "Seed=" +str(i)
# mixture, post, ll = naive_em.run(X, mixture, post)
# print("EM output: ")
# common.plot(X, mixture, post, title)
# print(i, ll)
# mixture, post = common.init(X, K, i)
# mixture, post, cost = kmeans.run(X, mixture, post)
# print("K_means output: ")
# print(i, cost)
# title = "K-means Model with K=" + str(K) + " and " + "Seed=" + str(i)
# common.plot(X, mixture, post, title)

# mixture, post = common.init(X, 3, 0)
# print(naive_em.estep(X, mixture))
# for i in range(0, 5):
#     mixture, post = common.init(X, 12, i)
#     mixture, post, ll = em.run(X, mixture, post)
#     print(i, ll)
mixture, post = common.init(X, 12, 1)
mixture, post, ll = em.run(X, mixture, post)
X_pred = em.fill_matrix(X_gold, mixture)
n, d = X.shape
error = 0
count = 0
for i in range(n):
    for j in range(d):
        if X[i][j] == 0:
            error = error + (X_gold[i, j] - X_pred[i, j])**2
            count = count + 1
print(np.sqrt(error/count))