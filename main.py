import numpy as np
import matplotlib.pyplot as plt
from Sack import SackItem
import ga


def main():

    # entradas
    saco = SackItem('saco de dormir', weight=15, points=15)
    corda = SackItem('corda', 3, 10)
    canivete = SackItem('canivete', 2, 10)
    tocha = SackItem('tocha', 5, 5)
    garrafa = SackItem('garrafa', 9, 8)
    comida = SackItem('comida', 20, 17)

    itens = [saco, corda, canivete, tocha, garrafa, comida]
    # número de pesos
    num_weights = 6

    sol_per_pop = 8

    # população tem sol_per_pop cromossomos com num_weights gens
    pop_size = (sol_per_pop, num_weights)

    # Algoritmo genético
    num_generations = 100
    num_parents_mating = 4

    # geracao que foi atingida o numero maximo (43)
    TRIALS = 100
    r_max_val = []

    for _ in range(TRIALS):
        # População inicial
        new_pop = np.random.randint(2, size=pop_size)

        for generation in range(num_generations):
            print(f"Geração: {generation}")

            # medir o ‘fitness’ de cada cromossomo na população
            fitness = ga.cal_pop_fitness(itens, new_pop)

            if any(value == 43 for value in fitness):
                r_max_val.append(generation)
                break

            # Selecionar os melhores parents na população para o cross
            parents = ga.select_mating_pool(
                new_pop, fitness, num_parents_mating)

            # formar a próxima geração usando crossover
            offspring_crossover = ga.crossover(parents, offspring_size=(
                pop_size[0] - parents.shape[0], num_weights
            ))

            # adicionar variações children usando mutação
            offspring_mutation = ga.mutation(offspring_crossover)

            # criar a nova população baseada nos parents e children
            new_pop[0:parents.shape[0], :] = parents
            new_pop[parents.shape[0]:, :] = offspring_mutation

            fitness = ga.cal_pop_fitness(itens, new_pop)
            best_match_idx = (np.where(fitness == np.max(fitness)))

    print('media', np.mean(r_max_val))
    print('mediana', np.median(r_max_val))
    print('desvio padrao', np.std(r_max_val))

    plt.hist(r_max_val)
    plt.show()


if __name__ == '__main__':
    main()
