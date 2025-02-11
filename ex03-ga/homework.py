import random
import numpy as np

# Матрица расстояний между городами
distance_matrix = [
    [0, 10, 15, 20, 25, 30],
    [10, 0, 35, 25, 30, 45],
    [15, 35, 0, 30, 35, 50],
    [20, 25, 30, 0, 15, 30],
    [25, 30, 35, 15, 0, 15],
    [30, 45, 50, 30, 15, 0]
]

# Параметры генетического алгоритма
# количество маршрутов в популяции.
population_size = 100
# количество поколений, через которые будет проходить алгоритм.
generations = 1000
# вероятность мутации каждого гена (города) в маршруте.
mutation_rate = 0.01

# Инициализация популяции
def initialize_population(size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(size)]

# Оценка приспособленности
def calculate_fitness(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    # добавляет расстояние от последнего города до первого, замыкая маршрут
    total_distance += distance_matrix[route[-1]][route[0]]
    return total_distance

# Селекция выбирает лучшие маршруты из популяции
def selection(population, distance_matrix):
    fitness_scores = [(route, calculate_fitness(route, distance_matrix)) for route in population]
    # сортирует маршруты по длине.
    fitness_scores.sort(key=lambda x: x[1])
    # возвращает половину лучших маршрутов.
    return [route for route, _ in fitness_scores[:population_size // 2]]

# Скрещивание (кроссовер)
def crossover(parent1, parent2):
    #  выбирает два случайных индекса.
    start, end = sorted(random.sample(range(len(parent1)), 2))
    # копирует участок из первого родителя.
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    parent2_index = 0
    # заполняет оставшиеся позиции городами из второго родителя, избегая дублирования.
    for i in range(len(child)):
        if child[i] is None:
            while parent2[parent2_index] in child:
                parent2_index += 1
            child[i] = parent2[parent2_index]
    return child

# Мутация
def mutate(route, mutation_rate):
    for i in range(len(route)):
        # проверяет, нужно ли выполнить мутацию
        if random.random() < mutation_rate:
            # выбирает случайный индекс для обмена
            swap_with = int(random.random() * len(route))
            # меняет местами города
            route[i], route[swap_with] = route[swap_with], route[i]
    return route

# Генетический алгоритм
def genetic_algorithm(distance_matrix, population_size, generations, mutation_rate):
    num_cities = len(distance_matrix)
    population = initialize_population(population_size, num_cities)
    best_route = None
    best_distance = float('inf')

    # проходит по каждому поколению
    for generation in range(generations):
        # выполняет селекцию
        population = selection(population, distance_matrix)
        new_population = []

        # гарантирует, что размер популяции четный
        if population_size % 2 != 0:
            population_size += 1

        # проходит по парам родителе
        for i in range(0, population_size, 2):
            if i + 1 < len(population):
                parent1 = population[i]
                parent2 = population[i + 1]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                # добавляет потомков в новую популяцию после мутации
                new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = new_population

        # проходит по каждому маршруту в популяции
        for route in population:
            # вычисляет длину маршрута
            distance = calculate_fitness(route, distance_matrix)
            # проверяет, является ли текущий маршрут лучшим
            if distance < best_distance:
                best_distance = distance
                best_route = route

        #  выводит текущее лучшее расстояние каждые 100 поколений.
        if generation % 100 == 0:
            print(f"Генерация {generation}: Лучшее расстояние = {best_distance}")

    return best_route, best_distance

# Запуск генетического алгоритма
best_route, best_distance = genetic_algorithm(distance_matrix, population_size, generations, mutation_rate)
print(f"Лучший путь: {best_route}")
print(f"Лучшее расстояние: {best_distance}")
