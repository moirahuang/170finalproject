import networkx as nx
# import cvxpy
import numpy as np
import os
import heapq
import math
import random
from simanneal import Annealer

###########################################
# Change this variable to the path to
# the folder containing all three input
# size category folders
###########################################
path_to_inputs = "./all_inputs"

###########################################
# Change this variable if you want
# your outputs to be put in a
# different folder
###########################################
path_to_outputs = "./outputs"

def parse_input(folder_name):
    '''
        Parses an input and returns the corresponding graph and parameters

        Inputs:
            folder_name - a string representing the path to the input folder

        Outputs:
            (graph, num_buses, size_bus, constraints)
            graph - the graph as a NetworkX object
            num_buses - an integer representing the number of buses you can allocate to
            size_buses - an integer representing the number of students that can fit on a bus
            constraints - a list where each element is a list vertices which represents a single rowdy group
    '''
    graph = nx.read_gml(folder_name + "/graph.gml")
    parameters = open(folder_name + "/parameters.txt")
    num_buses = int(parameters.readline())
    size_bus = int(parameters.readline())
    constraints = []

    for line in parameters:
        line = line[1: -2]
        curr_constraint = [num.replace("'", "") for num in line.split(", ")]
        constraints.append(curr_constraint)

    return graph, num_buses, size_bus, constraints

class SimulatedAnnealer(Annealer):

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, heuristic_matrix):
        self.heuristic_matrix = heuristic_matrix
        super(SimulatedAnnealer, self).__init__(state)  # important!

    def move(self):
        """Swaps two friends in bus."""
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        """Calculates the length of the route."""
        e = 0
        for i in range(len(self.state)):
            e += self.heuristic_matrix[self.state[i-1]][self.state[i]]
        return e

def dict_to_string(dict):
    result = ""
    for key in dict:
        if len(dict[key]) != 0:
            result = result + str(dict[key]) + "\n"
        else:
            result = result + "[]" + "\n"
    return result

def solve(graph, num_buses, size_bus, constraints):
    #TODO: Write this method as you like. We'd recommend changing the arguments here as well
    # graph, num_buses, size_bus, constraints = parse_input(path_to_outputs)

    # construct a dictionary mapping each person to the rowdy groups they're in

    student_names = list(graph.nodes())
    name_to_idx = dict()

    for i, student in enumerate(student_names):
        name_to_idx[student] = i

    num_rowdy_groups = len(constraints)
    bus_assignments = {}
    for i in range(num_buses):
        bus_assignments[i] = []
    #M
    rowdy_group_to_students = np.zeros((len(constraints), len(student_names)))
    for rg_index in range(len(constraints)):
        for student_string in student_names:
            if student_string in constraints[rg_index]:
                rowdy_group_to_students[rg_index, name_to_idx[student_string]] = 1

    sums = np.sum(rowdy_group_to_students, axis=1)

    scaled_rowdy_group_to_students = rowdy_group_to_students / sums[:,None]
    #C
    fraction_of_rowdy_group_in_bus = np.zeros(shape=(num_buses, num_rowdy_groups), dtype=float)
    #L
    number_of_friendships_in_bus_for_rowdy_group = np.zeros(shape=(num_buses, num_rowdy_groups), dtype=float)
    #sort students by number of rowdy groups they're in
    # iterate through every person
    student_ordering = np.argsort(-np.sum(rowdy_group_to_students, axis=0))
    for i, student_idx in enumerate(student_ordering[:num_buses]):
        update_data(student_idx, i, student_names, rowdy_group_to_students, np.zeros(shape=(num_buses, num_rowdy_groups)), fraction_of_rowdy_group_in_bus, number_of_friendships_in_bus_for_rowdy_group,
                        bus_assignments, scaled_rowdy_group_to_students)

    for student_idx in student_ordering[num_buses:]:
        num_friends_in_rg = np.zeros(shape=(num_rowdy_groups))

        additional_friendships = np.zeros(shape=(num_buses, 1), dtype=float)
        friend_count_in_rgs = np.zeros(shape=(num_buses, num_rowdy_groups))

        for bus in range(num_buses):
            count = 0
            for friend in graph.adj[student_names[student_idx]]:
                if name_to_idx[friend] in bus_assignments[bus]:
                    friend_rg = rowdy_group_to_students[:,name_to_idx[friend]]
                    student_rg = rowdy_group_to_students[:,student_idx]
                    common_rgs = np.where(friend_rg + student_rg == 2)
                    for common_rg in common_rgs:
                        friend_count_in_rgs[bus, common_rg] += 1
                    count+=1
            additional_friendships[bus] = count

        number_of_friendships_in_bus_for_rowdy_group_temp = number_of_friendships_in_bus_for_rowdy_group + additional_friendships @ rowdy_group_to_students[:,student_idx].reshape(1,num_rowdy_groups)
        fraction_of_rowdy_group_in_bus_temp               = fraction_of_rowdy_group_in_bus +  np.ones(shape=additional_friendships.shape) @ scaled_rowdy_group_to_students[:,student_idx].reshape(1, num_rowdy_groups)
        floored_fraction_of_rowdy_group_in_bus_temp       = np.floor(fraction_of_rowdy_group_in_bus_temp)

        reward_vector = additional_friendships.reshape(additional_friendships.size)
        cost_matrix = floored_fraction_of_rowdy_group_in_bus_temp + fraction_of_rowdy_group_in_bus_temp / num_rowdy_groups
        rowdy_groups_student_is_in = rowdy_group_to_students[:,student_idx]
        cost_vector = cost_matrix @ rowdy_groups_student_is_in
        heuristics = reward_vector - cost_vector
        sorted_heuristic_indices = np.argsort(-heuristics)

        for idx in sorted_heuristic_indices:
            load = len(bus_assignments[idx])
            if load < size_bus:
                update_data(student_idx, idx, student_names, rowdy_group_to_students, friend_count_in_rgs, fraction_of_rowdy_group_in_bus, number_of_friendships_in_bus_for_rowdy_group,
                                bus_assignments, scaled_rowdy_group_to_students)
                break

    first_pass = dict_to_string(bus_assignments)

    # create a distance matrix
    friendships = np.zeros(shape=(num_buses, len(student_names)))
    for bus_i in range(len(first_pass)):
        for student in first_pass[i]:
            for friend in graph.adj[student]:
                if friend in first_pass[i]:
                    count += 1
            friendships[bus_i][name_to_idx[student]] = count
    print(friendships)

    tsp = SimulatedAnnealer(first_pass, friendships)
    tsp.steps = 100000
    # since our state is just a list, slice is the fastest way to copy
    tsp.copy_strategy = "slice"
    state, e = tsp.anneal()

    return state
        #append friend to bus
    # heuristic is # of friendships created - factor * potential rowdy groups created (but make -large value if a rowdy group is created--recalculate friendships if a rowdy group has to be made)

    # simulated anealing to swap for best solution

# Update memoized data
def update_data(student_idx, bus_idx, student_names, rowdy_group_to_students, friend_count_in_rgs, fraction_of_rowdy_group_in_bus, number_of_friendships_in_bus_for_rowdy_group,
                bus_assignments, scaled_rowdy_group_to_students):
    bus_assignments[bus_idx].append(student_names[student_idx])
    rowdy_groups_student_is_in = rowdy_group_to_students[:, student_idx]
    fraction_of_rowdy_group_in_bus[bus_idx,:] += scaled_rowdy_group_to_students[:, student_idx]
    number_of_friendships_in_bus_for_rowdy_group += friend_count_in_rgs



def main():
    '''
        Main method which iterates over all inputs and calls `solve` on each.
        The student should modify `solve` to return their solution and modify
        the portion which writes it to a file to make sure their output is
        formatted correctly.
    '''
    size_categories = ["small"]
    if not os.path.isdir(path_to_outputs):
        os.mkdir(path_to_outputs)

    for size in size_categories:
        category_path = path_to_inputs + "/" + size
        output_category_path = path_to_outputs + "/" + size
        category_dir = os.fsencode(category_path)

        if not os.path.isdir(output_category_path):
            os.mkdir(output_category_path)

        for input_folder in os.listdir(category_dir):
            # input_name = os.fsdecode(input_folder)
            graph, num_buses, size_bus, constraints = parse_input(category_path + "/" + "15")
            solution = solve(graph, num_buses, size_bus, constraints)
            output_file = open(output_category_path + "/" + "15" + ".out", "w")

            #TODO: modify this to write your solution to your
            #      file properly as it might not be correct to
            #      just write the variable solution to a file
            output_file.write(solution)

            output_file.close()

if __name__ == '__main__':
    main()
