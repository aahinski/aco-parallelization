#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include <chrono>

std::vector<std::vector<double>> readCoordinates(const std::string& filename) {
    std::vector<std::vector<double>> coordinates;

    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return coordinates;
    }

    std::string line;
    int lineCount = 0;
    while (std::getline(file, line)) {
        if (lineCount > 9) {
            std::istringstream iss(line);
            double x, y;
            int id;
            if (iss >> id >> x >> y) {
                coordinates.push_back({ x, y });
            }
        }
        lineCount++;
    }

    return coordinates;
}

std::vector<std::vector<double>> get_distance_matrix(const std::vector<std::vector<double>>& coordinates) {
    int n = coordinates.size();
    std::vector<std::vector<double>> dist_matrix(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double dist = std::sqrt(std::pow(coordinates[i][0] - coordinates[j][0], 2) +
                std::pow(coordinates[i][1] - coordinates[j][1], 2));
            dist_matrix[i][j] = dist;
            dist_matrix[j][i] = dist;
        }
    }

    return dist_matrix;
}

double calculate_path_length(std::vector<int>& path, const std::vector<std::vector<double>>& distance_matrix) {
    double length = 0;

    for (int i = 0; i < path.size() - 1; ++i) {
        length += distance_matrix[path[i]][path[i + 1]];
    }

    length += distance_matrix[path.back()][path[0]];
    return length;
}

double probability(const std::vector<double>& pheromones, double heuristic, int current_point,
    int considerated_point, const std::vector<int>& unvisited, double a, double b, int n) {
    double numerator = std::pow(pheromones[current_point * n + considerated_point], a) *
        std::pow(heuristic, b);
    double denominator = 0.0;

    for (int l = 0; l < unvisited.size(); ++l) {
        denominator += std::pow(pheromones[current_point * n + l], a) * std::pow(heuristic, b);
    }

    return numerator / denominator;
}

std::vector<int> two_opt(std::vector<int>& path, const std::vector<std::vector<double>>& distance_matrix) {
    bool improved = true;

    while (improved) {
        improved = false;

        for (int i = 1; i < path.size() - 2; ++i) {
            for (int j = i + 1; j < path.size(); ++j) {
                if (j - i == 1) {
                    continue;
                }

                std::vector<int> new_path;
                new_path.assign(path.begin(), path.end());
                std::reverse(new_path.begin() + i, new_path.begin() + j);
                double new_length = calculate_path_length(new_path, distance_matrix);

                if (new_length < calculate_path_length(path, distance_matrix)) {
                    path.assign(new_path.begin(), new_path.end());
                    improved = true;
                }
            }
        }
    }

    return path;
}

std::pair<std::vector<int>, double> ant_colony_optimization_with_2opt(
    const std::vector<std::vector<double>>& distance_matrix,
    int iterations_count, int ants_count,
    double evaporaton_rate, double a, double b, int rank, int size) {
    int num_threads = 4;
    int n = distance_matrix.size();
    std::vector<double> pheromones(n * n, 1.0);

    std::vector<int> best_path(n);
    double best_path_length = std::numeric_limits<double>::infinity();

    for (int iteration = 0; iteration < iterations_count; ++iteration) {
        std::vector<int> paths;
        std::vector<double> path_lengths;

        int ants_per_process = ants_count / size;
        int start_ant = rank * ants_per_process;
        int end_ant = (rank == size - 1) ? ants_count : start_ant + ants_per_process;

        for (int ant = start_ant; ant < end_ant; ++ant) {
            int current_point = std::rand() % n;
            std::vector<bool> visited(n, false);
            visited[current_point] = true;
            std::vector<int> path = { current_point };
            double path_length = 0.0;
            while (std::find(visited.begin(), visited.end(), false) != visited.end()) {
                std::vector<int> unvisited;
                for (int i = 0; i < n; ++i) {
                    if (!visited[i]) {
                        unvisited.push_back(i);
                    }
                }

                std::vector<double> probabilities(unvisited.size());
                for (int i = 0; i < unvisited.size(); ++i) {
                    probabilities[i] = probability(pheromones, 1 / distance_matrix[current_point][unvisited[i]],
                        current_point, unvisited[i], unvisited, a, b, n);
                }

                int next_point = unvisited[std::rand() % unvisited.size()];
                path.push_back(next_point);
                visited[next_point] = true;
                current_point = next_point;
            }

            path = two_opt(path, distance_matrix);
            for (int i = 0; i < n; i++) {
                path_length = calculate_path_length(path, distance_matrix);
                paths.push_back(path[i]);
            }
            path_lengths.push_back(path_length);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> all_paths;
        std::vector<double> all_path_lengths;

        if (rank == 0) {
            all_paths.resize(ants_count * n);
            all_path_lengths.resize(ants_count);
        }

        MPI_Gather(paths.data(), ants_per_process * n, MPI_INT, all_paths.data(), ants_per_process * n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(path_lengths.data(), ants_per_process, MPI_DOUBLE, all_path_lengths.data(), ants_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    pheromones[i * n + j] *= (1 - evaporaton_rate);
                }
            }

            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < ants_count; ++i) {
                double delta = 1 / all_path_lengths[i];
                for (int j = 0; j < (n - 1); ++j) {
                    pheromones[all_paths[n * i + j] * n + all_paths[n * i + (j + 1)]] += delta;
                }
                pheromones[all_paths[n * i + (n - 1)] * n + all_paths[n * i]] += delta;
            }

            int best_path_index = 0;
            for (int i = 0; i < ants_count; ++i) {
                if (all_path_lengths[i] < best_path_length) {
                    best_path_length = all_path_lengths[i];
                    best_path_index = i;
                }
            }

            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < n; i++) {
                best_path[i] = all_paths[n * best_path_index + i];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(pheromones.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    return std::make_pair(best_path, best_path_length);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::srand(rank);

    std::string filename = "C:/Users/ahins/BSU/sem 6/курсач/датасеты/qa194.tsp";
    std::vector<std::vector<double>> coordinates = readCoordinates(filename);

    std::vector<std::vector<double>> distance_matrix = get_distance_matrix(coordinates);

    auto start_time = std::chrono::high_resolution_clock::now();

    MPI_Barrier(MPI_COMM_WORLD);

    std::pair<std::vector<int>, double> result_with_2opt =
        ant_colony_optimization_with_2opt(distance_matrix, 10, 4, 0.8, 4, 10, rank, size);

    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (rank == 0) {
        std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
        for (int i = 0; i < distance_matrix.size(); ++i) {
            std::cout << result_with_2opt.first[i] << ", ";
        }
        std::cout << std::endl << "Best path length: " << result_with_2opt.second;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
