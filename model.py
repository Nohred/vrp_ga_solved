import random
import time
from VRParser import VRParser
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')

# Problem Name: A045-03f
# Problem Type: ACVRP
# Number of Nodes (Dimension): 45
# Edge Weight Type: EXPLICIT
# Edge Weight Format: FULL_MATRIX
# Depot: 45


class GeneticSolver:
    def __init__(self, parser, population_size=1000, generations=10000, mutation_rate=0.5, elite_size=5):
        self.parser = parser
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = []

    def get_random_individual(self):
        nodes = list(range(0, self.parser.dimension - 1))  # 0 to 43 (excluye depósito)
        random.shuffle(nodes)
        routes = []

        while nodes:
            route_length = random.randint(1, min(len(nodes), 10))
            route = nodes[:route_length]
            nodes = nodes[route_length:]
            routes.append(route) 

        return routes

    def evaluate_fitness(self, individual, print_penalty=False):
        total_distance = 0
        visited = []
        depot = self.parser.depot - 1  # índice 44
        penalty = 1

        for route in individual:
            # Calcula distancia desde el depósito al primer nodo
            total_distance += self.parser.edge_weights[depot][route[0]]

            # Distancia entre nodos internos
            for i in range(len(route) - 1):
                total_distance += self.parser.edge_weights[route[i]][route[i + 1]]

            # Distancia del último nodo al depósito (si aplica)
            if self.parser.return_to_depot:
                total_distance += self.parser.edge_weights[route[-1]][depot]

            visited.extend(route)

        unique_visited = set(visited)
        total_nodes = self.parser.dimension - 1

        # Penalización si no se visitan todos los nodos
        nodes_not_visited = total_nodes - len(unique_visited)
        penalty += 1 * nodes_not_visited

        # Penalización si algún nodo se visita más de una vez
        nodes_repeated = len(visited) - len(unique_visited)
        penalty += 1 * nodes_repeated

        # n_routes_penalty = len(individual) * 1000
        n_routes_penalty = 0

        if print_penalty:
            print(f"Penalty por visitar un nodos varias veces = {nodes_repeated}")
            print(f"Penalty por no visitar todos los nodos = {nodes_not_visited}")
            print(f"Penalty por rutas = {n_routes_penalty}")
            return total_distance

        return total_distance  * penalty + n_routes_penalty

    def initialize_population(self):
        self.population = [self.get_random_individual() for _ in range(self.population_size)]

    def selection(self):
        sorted_population = sorted(self.population, key=self.evaluate_fitness)
        return sorted_population[:self.elite_size]

    def crossover(self, parent1, parent2):
        # 1. Copiar una ruta completa al azar desde uno de los padres
        child_nodes = set()
        child = []

        # Elige rutas de ambos padres
        p1_routes = parent1.copy()
        p2_routes = parent2.copy()
        random.shuffle(p1_routes)
        random.shuffle(p2_routes)

        # Agrega rutas únicas desde padre 1 mientras no se repitan nodos
        for route in p1_routes:
            if any(node in child_nodes for node in route):
                continue
            child.append(route.copy())
            child_nodes.update(route)

        # Completa con nodos no usados del padre 2
        remaining_nodes = [node for route in p2_routes for node in route if node not in child_nodes]
        random.shuffle(remaining_nodes)

        while remaining_nodes:
            length = random.randint(2, min(len(remaining_nodes), 8))
            route = remaining_nodes[:length]
            remaining_nodes = remaining_nodes[length:]
            child.append(route)
            child_nodes.update(route)

        return child

    def crossover_1(self, parent1, parent2):
        """
        Crossover alternativo: mezcla nodos de ambos padres (en orden relativo),
        eliminando duplicados y reagrupando en nuevas rutas.
        """

        # 1. Aplanar ambos padres en una lista secuencial
        flat_p1 = [node for route in parent1 for node in route]
        flat_p2 = [node for route in parent2 for node in route]

        # 2. Combinar alternando nodos, respetando el primero visto
        combined = []
        seen = set()
        for n1, n2 in zip(flat_p1, flat_p2):
            if n1 not in seen:
                combined.append(n1)
                seen.add(n1)
            if n2 not in seen:
                combined.append(n2)
                seen.add(n2)

        # 3. Añadir nodos restantes si las listas eran de distinta longitud
        for node in flat_p1 + flat_p2:
            if node not in seen:
                combined.append(node)
                seen.add(node)

        # 4. Reagrupar nodos en rutas aleatorias (2 a 8 nodos), con protección
        random.shuffle(combined)
        child = []
        while combined:
            if len(combined) >= 2:
                length = random.randint(2, min(len(combined), 8))
            else:
                length = 1
            route = combined[:length]
            combined = combined[length:]
            child.append(route)

        return child

    def crossover_pmx(self, parent1, parent2):
        # 1. Aplana todos los nodos en una lista única
        p1 = [node for route in parent1 for node in route]
        p2 = [node for route in parent2 for node in route]

        # 2. PMX: elige dos puntos de cruce
        size = len(p1)
        cx_point1 = random.randint(0, size - 2)
        cx_point2 = random.randint(cx_point1 + 1, size - 1)

        # 3. Crear hijo con None
        child = [None] * size

        # 4. Copiar segmento fijo del padre 1
        child[cx_point1:cx_point2 + 1] = p1[cx_point1:cx_point2 + 1]

        # 5. Mapeo entre genes
        for i in range(cx_point1, cx_point2 + 1):
            if p2[i] not in child:
                val = p2[i]
                pos = i
                while True:
                    mapped_val = p1[pos]
                    if mapped_val in child:
                        pos = p2.index(mapped_val)
                    else:
                        break
                child[pos] = val

        # 6. Rellenar lo demás del padre 2
        for i in range(size):
            if child[i] is None:
                child[i] = p2[i]

        # 7. Redistribuir en rutas aleatorias (2 a 8 nodos)
        random.shuffle(child)
        new_routes = []

        while child:
            if len(child) < 2:
                new_routes.append(child)
                break
            length = random.randint(2, min(len(child), 8))
            route = child[:length]
            child = child[length:]
            new_routes.append(route)

        return new_routes

    def mutate(self, individual):
        for _ in range(random.randint(1, 3)):  # Realizar varias pequeñas mutaciones
            route_idx = random.randint(0, len(individual) - 1)
            route = individual[route_idx]
            if len(route) >= 2 and random.random() < self.mutation_rate:
                i, j = random.sample(range(len(route)), 2)
                route[i], route[j] = route[j], route[i]
            elif len(route) >= 1 and len(individual) > 1:
                # mover nodo a otra ruta
                target_idx = random.choice([i for i in range(len(individual)) if i != route_idx])
                node = route.pop(random.randint(0, len(route) - 1))
                individual[target_idx].insert(random.randint(0, len(individual[target_idx])), node)
                if not route:
                    individual.remove(route)

    def run(self):
        self.initialize_population()
        history = []
        for generation in range(self.generations):
            elites = self.selection()
            new_population = elites[:]

            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(elites, 2)
                child = self.crossover(parent1, parent2)
                # child = self.crossover_pmx(parent1,parent2)
                self.mutate(child)
                new_population.append(child)

            self.population = new_population

            if generation % 10 == 0:
                best = min(self.population, key=self.evaluate_fitness)
                # eval_real = self.evaluate_fitness(best) - len(best)*1000
                eval_real = self.evaluate_fitness(best)
                print(f"Gen {generation} - Best Fitness: {eval_real}")
                history.append(eval_real)
            

        best_solution = min(self.population, key=self.evaluate_fitness)
        print("Best Solution Found:")
        for route in best_solution:
            full_route = [self.parser.depot - 1] + route + [self.parser.depot - 1]
            print(full_route)

        return history, best_solution

def plot_fitness_history(history):
    plt.figure(figsize=(10, 5))
    generations = [i * 10 for i in range(len(history))]
    plt.plot(generations, history, label="Fitness", color='blue', linewidth=2)
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Evolución del Fitness - WOC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def test_configuration(config, runs=10):
    results = []
    for i in range(runs):
        seed = i if config.get('fixed_seed', False) else random.randint(0, 10000)
        random.seed(seed)

        solver = GeneticSolver(
            parser=config["parser"],
            population_size=config["population_size"],
            generations=config["generations"],
            mutation_rate=config["mutation_rate"],
            elite_size=config["elite_size"]
        )

        if config["crossover"] == "pmx":
            solver.crossover = solver.crossover_pmx
        start = time.time()
        history, best = solver.run()
        end = time.time()
        # fitness = solver.evaluate_fitness(best) - len(best)*1000
        fitness = solver.evaluate_fitness(best)
        results.append({
            "seed": seed,
            "best_fitness": fitness,
            "final_generation": len(history)*10,
            "time": end-start
        })

    df = pd.DataFrame(results)
    summary = df.describe().loc[["mean", "std", "min", "max"]]
    print(summary)
    return df, summary


if __name__ == "__main__":
    random.seed(42)
    path = 'data/A045-03f.dat'
    data = VRParser(path)
    solver = GeneticSolver(data, population_size=100, generations=10000, mutation_rate=0.5, elite_size=5)

    print("\n#####################################################")
    print("Solving...")
    start = time.time()
    history, best = solver.run()
    end = time.time()
    print("Best Individual Fitness: ", solver.evaluate_fitness(best, print_penalty=True))
    print(f"Tiempo: {end-start} segundos")
    plot_fitness_history(history)
    
# if __name__ == "__main__":
    # configA = {
    # "parser": VRParser("data/A045-03f.dat"),
    # "population_size": 100,
    # "generations": 5000,
    # "mutation_rate": 0.5,
    # "elite_size": 5,
    # "crossover": "standard",
    # "fixed_seed": 42
    # }
    # configB = {
    # "parser": VRParser("data/A045-03f.dat"),
    # "population_size": 100,
    # "generations": 5000,
    # "mutation_rate": 0.5,
    # "elite_size": 5,
    # "crossover": "pmx", 
    # "fixed_seed": 42
    # }
    # configC = {
    # "parser": VRParser("data/A045-03f.dat"),
    # "population_size": 100,
    # "generations": 10000,
    # "mutation_rate": 0.5,
    # "elite_size": 5,
    # "crossover": "standard",
    # "fixed_seed": 42
    # }
    # configD = {
    # "parser": VRParser("data/A045-03f.dat"),
    # "population_size": 500,
    # "generations": 5000,
    # "mutation_rate": 0.5,
    # "elite_size": 5,
    # "crossover": "standard",
    # "fixed_seed": 42
    # }

#     dfA, summaryA = test_configuration(configA, runs=10)
#     dfB, summaryB = test_configuration(configA, runs=10)
#     dfC, summaryC = test_configuration(configC, runs=10)
#     dfD, summaryD = test_configuration(configD, runs=10)

#     print("Prueba A\n")
#     print(dfA)
#     print(summaryA)
#     print("\nPrueba B\n")
#     print(dfB)
#     print(summaryB)
#     print("\nPrueba C\n")
#     print(dfC)
#     print(summaryC)
#     print("\nPrueba D\n")
#     print(dfD)
#     print(summaryD)





    