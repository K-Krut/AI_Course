import random
import numpy as np
import numpy.random as npr


class Chromosome:
    MUTATION_RATE = 0.1

    def __init__(self, min_value, max_value, value=None, mut_rate=0.1):
        self.fitness = 0
        self.min_value = min_value
        self.max_value = max_value
        self.value = random.uniform(min_value, max_value) if value is None else value
        self.fitness = self.fitness_function(self.value)
        self.mutation_rate = mut_rate

    def get_value(self):
        return self.value

    def get_fitness(self):
        return self.fitness

    @staticmethod
    def fitness_function(x):
        return 5 * np.sin(10 * x) * np.sin(3 * x) / x ** x

    def mutate(self):
        rate = random.uniform(0, 1)
        if rate < self.mutation_rate:
            delta = self.get_sign() * 0.1
            if self.min_value <= self.value + delta <= self.max_value:
                self.value += delta
                self.fitness = self.fitness_function(self.value)

    @staticmethod
    def get_sign():
        sign = random.uniform(0, 1) * 2 - 1
        return 1 if sign >= 0 else -1

    def __str__(self):
        return f"Chromosome{{value={self.value}, fitness={self.fitness}}}"


class Population:

    def __init__(self, population_n, reversed, min_value, max_value):
        self.population_n = population_n
        self.reversed = reversed
        self.population = [Chromosome(min_value, max_value) for _ in range(population_n)]
        self.min_value = min_value
        self.max_value = max_value
        self.ranging_rate = 0.125

    def sort_chromosomes(self):
        self.population = sorted(self.population, reverse=self.reversed, key=lambda x: x.fitness)

    def get_min_parents(self, index):
        return min(self.population[index].value, self.population[index + 1].value)

    def get_max_parents(self, index):
        return max(self.population[index].value, self.population[index + 1].value)

    def check_range(self, value, range='min'):
        return eval(f'self.{range}_value if {value} {"<" if range == "min" else ">"} self.{range}_value else {value}')

    def check_min(self, value):
        return self.min_value if value < self.min_value else value

    def check_max(self, value):
        return self.max_value if value > self.max_value else value

    # get ranges for children
    def get_ranges(self, index):
        minVal = self.get_min_parents(index)
        maxVal = self.get_max_parents(index)
        minRange = minVal - self.ranging_rate * (maxVal - minVal)
        maxRange = maxVal + self.ranging_rate * (maxVal - minVal)
        minRange = self.check_min(minRange)
        maxRange = self.check_max(maxRange)

        return minRange, maxRange

    # random value for children
    @staticmethod
    def rand_value(min_range, max_range):
        return npr.random() * (max_range - min_range) + min_range

    def crossover(self):
        children = []
        for i in range(0, self.population_n, 2):
            ranges = self.get_ranges(i)
            child1 = Chromosome(ranges[0], ranges[1], self.rand_value(ranges[0], ranges[1]))
            child2 = Chromosome(ranges[0], ranges[1], self.rand_value(ranges[0], ranges[1]))

            children.append(child1)
            children.append(child2)
        self.population.extend(children)
        self.sort_chromosomes()
        self.population = self.population[:self.population_n]

    def mutate(self):
        for i in range(self.population_n):
            self.population[i].mutate()

    def getPopulation(self):
        return self.population

    def __str__(self):
        return f"Population{{populationNumber={self.population_n}, population={self.population}}}"


if __name__ == "__main__":
    population = Population(40, True, 0, 8)
    for i in range(50):
        population.sort_chromosomes()
        population.crossover()
        population.mutate()
    population.sort_chromosomes()
    print(str(population.getPopulation()[0].get_value()) + " " + str(population.getPopulation()[0].get_fitness()))

    population = Population(40, False, 0, 8)
    for i in range(50):
        population.sort_chromosomes()
        population.crossover()
        population.mutate()
    population.sort_chromosomes()
    print(str(population.getPopulation()[0].get_value()) + " " + str(population.getPopulation()[0].get_fitness()))