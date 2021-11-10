import numpy as np
import kmeans
import common
import naive_em
import em
import matplotlib.pyplot as plt

X = np.loadtxt("toy_data.txt")
# TODO: Your code here

K = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]
final_answer = []

for k in K:
    costs = []
    for i in range(len(seeds)):
        print("K = ", k, "and Seed = ", i)
        mixture, post = common.init(X, k, i)
        mixture, post, cost = kmeans.run(X, mixture, post)
        costs.append(cost)
    print("All costs for K = ", k, ": ", costs)
    print("min of All costs for K = ", k, ": ", np.min(costs))
    final_answer.append(np.min(costs))
    title = "K-means for K={}, seed={}, cost={}".format(k, i, np.min(costs))
    print(title)
    common.plot(X, mixture, post, title, key = 0)

print("FINAL ANSWER: ", final_answer)


################################################
K = 3
seed = 0
#expected log likelihood result = -1388.0818
mixture, post = common.init(X, K, seed=0)
post, ll = naive_em.estep(X, mixture)
print("Log-likelihood: {}".format(ll))

gaussian_mix = naive_em.mstep(X,post)


#################################################

K = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]
final_answer = []

for k in K:
    lls = []
    for i in range(len(seeds)):
        print("K = ", k, "and Seed = ", i)
        mixture, post = common.init(X, k, i)
        mixture, post, ll = naive_em.run(X, mixture, post)
        lls.append(ll)
    print("All lls for K = ", k, ": ", lls)
    print("min of All lls for K = ", k, ": ", np.min(lls))
    final_answer.append(np.min(lls))
    title = "EM for K={}, seed={}, ll={}".format(k, i, np.min(lls))
    print(title)
    common.plot(X, mixture, post, title, key = 1)

print("FINAL ANSWER: ", final_answer)

################################################

K = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]
final_answer_bics = []
final_answer_lls = []
final_answer = []
for k in K:
    lls = []
    bics = []
    for i in range(len(seeds)):
        print("K = ", k, "and Seed = ", i)
        mixture, post = common.init(X, k, i)
        mixture, post, ll = naive_em.run(X, mixture, post)
        lls.append(ll)
    print("All lls for K = ", k, ": ", lls)
    print("min of All lls for K = ", k, ": ", np.min(lls))
    final_answer_lls.append(np.min(lls))
    bic = common.bic(X, mixture, ll)
    print("BIC: ", bic)
    bics.append(bic)
    final_answer_bics.append(np.max(bics))
    title = "EM for K={}, seed={}, ll={}, BIC={}".format(k, i, np.min(lls), np.max(bics))
    print(title)
    common.plot(X, mixture, post, title, key = 1)
    final_answer.append(np.max(bics))

print("FINAL ANSWER: ", final_answer_bics)
print(np.max(final_answer_bics))


#################################################
X = np.loadtxt("netflix_incomplete.txt")

K = [1, 12]
seeds = [0, 1, 2, 3, 4]
final_answer = []

for k in K:
    lls = []
    for i in range(len(seeds)):
        print("K = ", k, "and Seed = ", i)
        mixture, post = common.init(X, k, i)
        mixture, post, ll = em.run(X, mixture, post)
        print("ll for K = ", k, ": ", ll)
        lls.append(ll)
    print("All lls: ", lls)
    print("max of lls: ", np.max(lls))
    final_answer.append(np.max(lls))
    title = "EM for K={}, seed={}, ll={}".format(k, i, np.max(lls))
    print(title)
    #common.plot(X, mixture, post, title, key = 1)

print("FINAL ANSWER: ", final_answer)


#################################################
K = 12
seed = 1

X_gold = np.loadtxt('netflix_complete.txt')
mixture, post = common.init(X, K, seed)
mixture, post, ll = em.run(X, mixture, post)
X_pred = em.fill_matrix(X, mixture)
X_gold = np.loadtxt('netflix_complete.txt')
print("RMSE result:", common.rmse(X_gold, X_pred))