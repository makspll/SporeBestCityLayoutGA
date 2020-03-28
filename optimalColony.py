import random
from enum import Enum
import sys
import matplotlib.pyplot as plt
class GA:
    # cP - crossoverProbability, the probability of each new individual 
    # being created with crossover rather than just being picked 
    # from the previous population
    # mP - mutationProbability, the probability that an individual undergoes a mutation
    # popSize - number of individuals in each generation
    # tournamentSize - in performing tournament selection, how many random individuals should be sampled
    # chromosome length includes the main hall at position 0 but it will not be displayed at the end
    def __init__(self, cP,mP,popSize,tournamentSize):
        self.cP = cP
        self.mP = mP
        self.popSize = popSize
        self.chromosomeLen = 12 # standard field size in spore
        self.tournamentSize = tournamentSize
        self.links = []
        self.population = []
        self.setRandomPopulation()
        self.generation = 0
        self.noImprovGens = 0
        self.minHappiness = 0

    # pM - production modifier, each planet has a different modifier to spice production, you can find out what it is
    # by placing a single factory and reading how much spice per link it produces, this is the modifier
    # cM - cost modifier, each planet has different costs, but ratios between building types are the same
    # to find out your cM, divide the cost of a house on your planet by 1600
    # minHappiness - the minimum happiness of the solution
    # links is a list of edges, without duplicates (i.e. the coordinates in the adjacency matrix where connections are present but only from bottom left triangle)
    def findSolution(self,pM,cM,links,minHappiness):
        self.links = links
        self.minHappiness = minHappiness

        i = 1
        bestFitness2 = 999999
        bestFitness = -999999

        # evaluate the first population
        for p in self.population:
            self.evaluateIndividual(p)

        # perform evolution
        while not(self.converged(bestFitness2-bestFitness)):
            # find the best chromosome
            best = None
            bestFitness = -1
            for p in self.population:
                if p.fitness > bestFitness:
                    bestFitness = p.fitness
                    best = p

            print("gen: " + str(i) + ", best: " + str(best))
    
            # generate new population

            newPop = self.generateNewPopulation()

            # find the best in new population
            best2 = None
            bestFitness2 = -1
            for p in self.population:
                if p.fitness > bestFitness2:
                    bestFitness2 = p.fitness
                    best2 = p
            self.population = newPop
            i+=1
        prodPoints,happPoints,costPoints = evaluate(self.links,best.genes)
        print("Solution Found!")
        print("spice:" + str(prodPoints*pM))
        print("happiness:" + str(happPoints))
        print("cost:" + str(costPoints*cM))
        print("assignment:" + str(best))
        return best

        
    def generateNewPopulation(self):

        # evaluate the old population
        for p in self.population:
            self.evaluateIndividual(p)

        newPopulation = []

        while len(newPopulation) < self.popSize:
            # for each individual determine if he's a survivor from the previous population
            # or an offspring from 2 other survivors
            newChromosome = None
            if random.random() < self.cP:
                ind1 = self.selectIndividual()
                ind2 = self.selectIndividual()

                newChromosome = self.crossover(ind1,ind2)
            else:
                newChromosome = self.selectIndividual()

            # decide if the individual mutates
            if random.random() < self.mP:
                self.mutate(newChromosome)

            newPopulation.append(newChromosome)
        return newPopulation

    def converged(self,fitnessDelta):
        
        if fitnessDelta < 1:
            self.noImprovGens +=1
        else:
            self.noImprovGens = 0
        
        if self.noImprovGens >= 5:
            return True
        else:
            return False

    def setRandomPopulation(self):
        self.population = []

        buildings = [Building.H,Building.F,Building.E,Building._]
        for i in range(self.popSize):
            p = Chromosome([])
            p.genes = random.choices(buildings,k=self.chromosomeLen)
            p.genes[0] = Building.H # 0 is always the city hall, which behaves like a house

            self.population.append(p)

    def selectIndividual(self):
        # select K individuals randomly and pick the best one
        sample = random.choices(self.population,k=self.tournamentSize)
        
        best = None
        bestFitness = -sys.float_info.max
        for p in sample:
            if p.fitness > bestFitness:
                best = p
                bestFitness = p.fitness

        # then choose the best one, or apply stochastic methods
        return best

    def crossover(self,a,b):
        # we always keep city hall at the begginning
        newChromosome = Chromosome([])
        for i in range(self.chromosomeLen):
            if random.random() < 0.5:
                newChromosome.genes.append(a.genes[i])
            else:
                newChromosome.genes.append(b.genes[i])
        return newChromosome

    def mutate(self,a):
        p = 1 / self.chromosomeLen
        for g in a.genes:
            if random.random() < p:
                g = [Building.H,Building.F,Building.E,Building._][random.randint(0,2)]

    def evaluateIndividual(self,a):
            a.fitness = self.evaluationFunction(a)

    def evaluationFunction(self,a):
        productionPoints,happinessPoints,costPoints = evaluate(self.links,a.genes)
        
        negativeHappinessPen = -10000 if happinessPoints < self.minHappiness else 0 
        return negativeHappinessPen + ((50/(costPoints+1))**2) + ((productionPoints)**2) 


class Chromosome:
    def __init__(self,genes):
        self.fitness = 0
        self.genes = genes

    def __str__(self):
        return "fitness: "+str(round(self.fitness,2))+" ,genes: "+''.join([self.genes[i].name for i in range(1,len(self.genes))])

class Building(Enum):
    _ = 0 # empty field
    H = 1 # house
    F = 2 # factory
    E = 3 # entertainment building
    def cost(self):
        if self.value == 1:
            return 1600 
        elif self.value == 2:
            return 1200
        elif self.value ==3:
            return 800
        else:
            return 0
    def happiness(self):
        if self.value == 1:
            return 0
        elif self.value == 2:
            return -1
        elif self.value ==3:
            return 1
        else:
            return 0
# the program works with any adjacency matrix for any colony layout, but has the pre-made spore layouts saved as well


# given a list of links of the form (i,j) 0 <= i,j < N, 0 being the city hall, a building type list of length N of buildings
# return the production points and happiness points along with the cost, which can be evaluated with a production multiplier to get the total spice per hour
# assumes there are no duplicate links
def  evaluate(links,buildings):
    productionPoints = happinessPoints = costPoints = 0
    # evaluate each link
    for l in links:
        A = buildings[l[0]]
        B = buildings[l[1]]
        # workout the spice/happiness values from the link using the enum values
        # possible connections:
        # H to F = 1 * 2 = 2
        # H to E = 1 * 3 = 3
        # E to F = 3 * 2 = 6

        value = A.value * B.value

        # +1P
        if value == 2:
            productionPoints+=1
        # +1H
        elif value == 3:
            happinessPoints+=1
        # -1H
        elif value == 6:
            happinessPoints-=1
        # no effect
        else:
            continue

    # evaluate total building cost
    cost = sum([x.cost() for x in buildings]) - buildings[0].cost()
    happiness = sum([x.happiness() for x in buildings])

    # evaluate happiness from buildings themselves

    return (productionPoints,happinessPoints+happiness,cost)
        
if __name__ == "__main__":
    colony = [(0,2),(0,4),(0,7),(0,9),(0,10),(1,2),(1,10),(2,3),(3,4),(3,5),(4,5),(5,6),(6,7),(6,8),(7,8),(8,9),(9,10),(10,11)]
    #5 city hall connections
    homeworld1 = [(0,2),(0,5),(0,7),(0,11),(1,2),(2,3),(2,11),(4,5),(5,6),(5,7),(6,7),(7,8),(7,9),(9,11),(9,10),(10,11)]
    #4 city hall connections
    homeworld2 = [(0,2),(0,3),(0,8),(0,9),(1,11),(2,3),(2,9),(3,4),(3,5),(4,5),(5,6),(7,8),(8,9),(9,11),(10,11)]
    #2 city hall connections
    homeworld3 = [(0,8),(0,10),(1,2),(2,3),(3,5),(3,6),(4,5),(6,7),(6,8),(7,8),(8,10),(9,10),(10,11)]
    #5 city hall connections
    homeworld4 = [(0,2),(0,4),(0,6),(0,8),(0,10),(1,2),(2,3),(2,4),(3,5),(4,5),(4,6),(5,6),(6,7),(6,8),(7,8),(8,9),(8,10),(9,10),(10,11)]
    #4 city hall connetions
    homeworld5 = [(0,2),(0,5),(0,8),(0,9),(1,2),(2,3),(2,4),(3,4),(4,6),(4,5),(5,8),(6,7),(7,8),(8,9),(9,10),(9,11)]
    #5 city hall connections
    homeworld6 = [(0,2),(0,4),(0,5),(0,9),(0,10),(1,2),(2,3),(2,4),(3,4),(5,6),(6,7),(7,8),(7,9),(8,9),(9,10),(10,11)]
    
    ga = GA(0.8,0.09,1000,7) # pretty optimal parameters
    ga.findSolution(12,16,homeworld6,0) 

