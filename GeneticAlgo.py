import random
import scipy
import numpy as np
import matplotlib.pyplot as plt

'''  -------------------- GLOBALS  -------------------- '''
N = 30          # population size
G = 10          # number of generations
l = 20          # number of genes in genetic string
p_m = 0.033     # mutation probability
p_c = 0.6       # single-point crossover probability

'''  -------------------- INDIVIDUAL CLASS -------------------- '''

class Individual:

    # initializes individual's genes. Passed in parent genes are optional
    def __init__(self):
        self.genes = np.array([random.choice([0,1]) for i in range(0, l)])
        self.rTNF = 0.0         # sum of normalized fitness + all previous invididuals in the population's 

    # returns binary representation of the gene character array
    def geneValue(self):
        val = 0
        for i in range(0, len(self.genes)):
            if (self.genes[i] == 1):
                val += (2**(l-1-i))
        return val

    # returns fitness of the individual
    def fitness(self):
        return (self.geneValue() / (2.0**l))**10.0

    # returns how many 1 bits are in the gene array
    def popcount(self):
        return bin(self.geneValue()).count('1')

    # getter for gene array
    def get_genes(self):
        return self.genes

    def set_genes(self, geneArray):
        self.genes = geneArray

    # setter for rTNF
    def set_rTNF(self, normFit):
        self.rTNF = normFit

    # getter for rTNF
    def get_rTNF(self):
        return self.rTNF

''' -------------------- STATISTICS CLASS -------------------- '''

class Stats:

    # member functions overwrite passed-in data!
    def __init__(self, Pop=[], avgFit=0, bestFit=0, numCorrect=0):
        self.P = Pop
        self.averageFitness = avgFit
        self.bestIndivFitness = bestFit
        self.numCorrectBits = numCorrect

    def avgFitness(self):
        total = 0.0
        for i in range(0, len(self.P)):
            total += self.P[i].fitness()
        return total / (N*1.0)

    def bestIndividual(self):
        maxFitness = -1.0
        bestIndex = 0
        for i in range(0, len(self.P)):
            if (self.P[i].fitness() > maxFitness):
                bestIndex = i
                maxFitness = self.P[i].fitness()
        return self.P[bestIndex]

    def bestIndividualFitness(self):
        return self.bestIndividual().fitness()

    def bestCorrectBits(self):
        return self.bestIndividual().popcount()

    def avgCorrectBits(self):
        total = 0
        for i in range(len(self.P)):
            total += self.P[i].popcount()
        return total / (N * 1.0)


''' -------------------- GENERATION LOOP -------------------- '''

S = []          # list of Stats objects for each generation
for g in range(0, G-1):

    ''' -------------------- FITNESS CALCULATION -------------------- '''

    # if it's our first generation...
    if (g == 0):
        P = np.array([Individual() for i in range(0,N)])         # create size N population of individuals

    totalFitness = 0.0                                # running total of population's fitness
    totalNormFitness = 0.0                            # running total of population's normalized fitness
    normFitness = np.zeros([N])                       # normalized fitness values

    # go through every individual in population and set fitness values
    for i in range(0, len(P)):
        totalFitness += P[i].fitness()                          # add to the total fitness                     

    # now that we have totals, go back and set normalized fitness values
    for i in range(0, len(P)):
        normFitness[i] = (P[i].fitness() / totalFitness)        # store normalized fitness value
        totalNormFitness += normFitness[i]                      # add to the total norm Fitness
        P[i].set_rTNF(totalNormFitness)                         # set individual's rTNF value

    # if first generation, save stats
    if (g == 0):
        S.append(Stats(P))

    ''' -------------------- MATING PROCESS -------------------- '''

    # create new individual array for the offspring
    P_prime = []
    
    # since P is intrinsically sorted by rTNF values, first individual in P with rTNF > rand1 is the parent.
    def selectParent(P, randNum):
        for i in range(0, len(P)):
            if (P[i].get_rTNF() > randNum):
                return P[i]

    # mate the parents and produce two children with either copied or shared genes.
    def crossover(parent1, parent2):

        child1 = Individual()
        child2 = Individual()
        genes1 = []
        genes2 = []

        # crossover?
        randNum = random.random()
        if (randNum < p_c):

            cbit = random.choice(range(l))
            for i in range(0, cbit+1):
                genes1.append(parent1.get_genes()[i])
                genes2.append(parent2.get_genes()[i])
            for i in range(cbit+1, l):
                genes1.append(parent2.get_genes()[i])
                genes2.append(parent1.get_genes()[i])
        else:
            genes1 = parent1.get_genes()
            genes2 = parent2.get_genes()

        child1.set_genes(genes1)
        child2.set_genes(genes2)

        return [child1, child2]

    def mutation(child1, child2):

        genes1 = child1.get_genes()
        genes2 = child2.get_genes()

        for i in range(0, l):
            rand1 = random.random()
            rand2 = random.random()

            # child 1 possible mutation
            if (rand1 < p_m):
                if (genes1[i] == 0):
                    genes1[i] = 1
                else:
                    genes1[i] = 0

            # child 2 possible mutation       
            if (rand2 < p_m):
                if (genes2[i] == 0):
                    genes2[i] = 1
                else:
                    genes2[i] = 0
            
        child1.set_genes(genes1)
        child2.set_genes(genes2)

        return [child1, child2]


    for i in range(0, N/2):

        rand1 = random.random()
        rand2 = random.random()

        # select two distinct parents based on parent selection criteria
        parent1 = selectParent(P, rand1)
        parent2 = selectParent(P, rand2)
        while (parent2 == parent1):
            rand2 = random.random()
            parent2 = selectParent(P, rand2)

        # mate parents and produce 2 children, store them in P_prime
        children = crossover(parent1, parent2)
        children = mutation(children[0], children[1])
        P_prime.append(children[0])
        P_prime.append(children[1])

    # create stats object by passing in population to constructor, append it to S
    stats = Stats(P_prime)
    S.append(stats)

    # children become the new prospective parents
    P = P_prime
    
''' -------------------- PLOTTING DATA -------------------- '''

# get arrays of data
avgFitness = []
bestIndivFitness = []
avgCorrectBits = []
bestIndivCorrectBits = []
for i in range(len(S)):
    avgFitness.append(S[i].avgFitness())
    bestIndivFitness.append(S[i].bestIndividualFitness())
    avgCorrectBits.append(S[i].avgCorrectBits())
    bestIndivCorrectBits.append(S[i].bestCorrectBits())

# plot all of the data using subplots
fig, axs = plt.subplots(2,2,figsize=(15,15))
axs[0,0].plot(range(1,G+1), avgFitness)
axs[0,0].set(xlabel='Generation', ylabel='Average Fitness', title="Avg. Fitness vs. Generation")
axs[1,0].plot(range(1,G+1), bestIndivFitness)
axs[1,0].set(xlabel="Generation", ylabel="Best Individual\'s Fitness", title="Best Individual\'s Fitness vs. Generation")
axs[0,1].plot(range(1,G+1), avgCorrectBits)
axs[0,1].set(xlabel='Generation', ylabel='Average Correct Bits',title="Avg. Correct Bits vs. Generation")
axs[1,1].plot(range(1,G+1), bestIndivCorrectBits)
axs[1,1].set(xlabel='Generation', ylabel='Best Individual\'s Correct Bits',title="Best Individual\'s Correct Bits vs. Generation")
plt.subplots_adjust(hspace=0.35)
plt.show()