from __future__ import division
import numpy as np

from hmmlearn import hmm



def calculateLikelyHood(model, X):
    score = model.score(np.atleast_2d(X).T)

    print ("\n\n[CalculateLikelyHood]: ")
    print( "\nobservations:")
    for observation in list(map(lambda x: observations[x], X)):
        print( " ", observation)

    print ("\nlikelyhood:", np.exp(score))

def optimizeStates(model, X):
    Y = model.decode(np.atleast_2d(X).T)
    print("\n\n[OptimizeStates]:")
    print( "\nobservations:")
    for observation in list(map(lambda x: observations[x], X)):
        print (" ", observation)

    print ("\nstates:")
    for state in list(map(lambda x: states[x], Y[1])):
        print (" ", state)


states = ["Gold", "Silver", "Bronze"]
n_states = len(states)

observations = ["Ruby", "Pearl", "Coral", "Sapphire"]
n_observations = len(observations)


start_probability = np.array([0.3, 0.3, 0.4])

transition_probability = np.array([
    [0.1, 0.5, 0.4],
    [0.4, 0.2, 0.4],
    [0.5, 0.3, 0.2]
    ])

emission_probability = np.array([
    [0.4, 0.2, 0.2, 0.2],
    [0.25, 0.25, 0.25, 0.25],
    [0.33, 0.33, 0.33, 0]
])

model = hmm.MultinomialHMM(n_components=3)

# 直接指定pi: startProbability, A: transmationProbability 和B: emissionProbability

model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability


X1 = [0,1,2]
X2 = [0,0,0]

if __name__ == '__main__':
    calculateLikelyHood(model, X1)
    optimizeStates(model, X1)
    calculateLikelyHood(model, X2)
    optimizeStates(model, X2)




