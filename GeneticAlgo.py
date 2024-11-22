import random
from deap import base, creator, tools, algorithms

# Blood types
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
DEMAND = [50, 30, 40, 20, 60, 25, 10, 5]  # Example demand for each blood type
OPTIMAL_STOCK = [70, 50, 60, 40, 90, 35, 20, 10]  # Optimal stock levels
TRANSPORT_COST_PER_UNIT = 1.5  # Example transport cost per unit

# Step 1: Define the problem as a minimization
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
creator.create("Individual", list, fitness=creator.FitnessMin)

# Step 2: Create a toolbox
toolbox = base.Toolbox()
toolbox.register("attr_stock", random.randint, 0, 100)  # Stock levels between 0 and 100
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_stock, n=len(BLOOD_TYPES))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Step 3: Define the fitness function
def evaluate(individual):
    # Calculate expired blood bags (stock exceeding demand + safety margin)
    expired_bags = sum(max(0, stock - demand - 10) for stock, demand in zip(individual, DEMAND))
    
    # Calculate unmet demand (demand not met by stock)
    unmet_demand = sum(max(0, demand - stock) for stock, demand in zip(individual, DEMAND))
    
    # Calculate transportation cost (assume restocking requires transporting the difference from optimal)
    transport_cost = sum(abs(stock - optimal) * TRANSPORT_COST_PER_UNIT for stock, optimal in zip(individual, OPTIMAL_STOCK))
    
    # Fitness value to minimize
    fitness = expired_bags + unmet_demand + transport_cost
    return fitness,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)  # Mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection

# Step 4: Run the Genetic Algorithm
def main():
    random.seed(42)  # For reproducibility
    
    # Create a population of 50 individuals
    population = toolbox.population(n=50)
    
    # Evaluate the initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    
    # Statistics to monitor progress
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])  # Corrected: access the first element of the tuple
    stats.register("avg", lambda fits: sum(fits) / len(fits))  # Calculate average fitness correctly
    stats.register("min", min)
    
    # Run the algorithm
    final_population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.7,  # Crossover probability
        mutpb=0.2,  # Mutation probability
        ngen=50,    # Number of generations
        stats=stats,
        verbose=True
    )
    
    # Print the best solution
    best_ind = tools.selBest(final_population, k=1)[0]
    print("Best individual (stock levels):", best_ind)
    print("Best fitness (minimized value):", best_ind.fitness.values[0])

if __name__ == "__main__":
    main()
