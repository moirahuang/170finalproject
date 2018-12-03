from __future__ import print_function

import os
import random

import networkx as nx
# import cvxpy
import numpy as np
from simanneal import Annealer
from functools import reduce

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
    def __init__(self, bus_assignments, friendships_in_bus_for_student, fraction_of_rowdy_group_in_bus, rowdy_group_student_membership_matrix, name_to_index, student_names,  number_of_friendships_in_bus_for_rowdy_group, scaled_rowdy_group_student_membership_matrix, constraints, size_bus, graph):
        self.friendships_in_bus_for_student = friendships_in_bus_for_student
        self.fraction_of_rowdy_group_in_bus = fraction_of_rowdy_group_in_bus
        self.rowdy_group_student_membership_matrix = rowdy_group_student_membership_matrix
        self.name_to_index = name_to_index
        self.student_names = student_names
        self.number_of_friendships_in_bus_for_rowdy_group = number_of_friendships_in_bus_for_rowdy_group
        self.bus_assignments = bus_assignments
        self.student_assignments = {}
        for bus in self.bus_assignments:
            for student_name in self.bus_assignments[bus]:
                self.student_assignments[self.name_to_index[student_name]] = bus
        self.scaled_rowdy_group_student_membership_matrix = scaled_rowdy_group_student_membership_matrix
        self.constraints = constraints
        self.size_bus = size_bus
        self.graph = graph
        super(SimulatedAnnealer, self).__init__(bus_assignments)  # important!
        self.buses_filled_minimally = list(filter(lambda key: len(bus_assignments[key]) == 1, self.bus_assignments.keys()))
        self.buses_filled_maximally = list(filter(lambda key: len(bus_assignments[key]) == size_bus , self.bus_assignments.keys()))
        self.buses_not_filled_minimally = list(filter(lambda key: len(bus_assignments[key]) > 1, self.bus_assignments.keys()))
        self.buses_not_filled_maximally = list(filter(lambda key: len(bus_assignments[key]) < size_bus , self.bus_assignments.keys()))

        self.max_energy = np.sum([len(self.friendships_in_bus_for_student[self.name_to_index[student_name]]) for student_name in self.student_names])
        self.Tmax       = 3 * self.energy()

        self.actions = [self.transfer, self.swap, self.permutation]
        self.action_probabilities = [0.5, 0.3, 0.2]
        self.cummulative_action_probabilities = np.cumsum(self.action_probabilities)

    def move(self):
        action = self.actions[np.argmin(self.cummulative_action_probabilities >= np.random.sample())]
        action()

    def transfer(self):
        student_name = ''
        bus_sent_from = -1
        num_attempts = 10 # infinite loop catch in case all buses are minimally filled
        while student_name == '' and num_attempts > 0:
            student_name = self.student_names[np.random.randint(len(self.student_names))]
            bus_sent_from = self.student_assignments[self.name_to_index[student_name]]
            if bus_sent_from in self.buses_filled_minimally:
                student_name = ''
            num_attempts -= 1

        if student_name == '':
            return None

        if num_attempts >= 0:
            feasible_buses_to_accept_transfer = self.buses_not_filled_maximally
            if bus_sent_from in feasible_buses_to_accept_transfer:
                feasible_buses_to_accept_transfer = list(filter(lambda e: bus_sent_from != e, feasible_buses_to_accept_transfer))
            if len(feasible_buses_to_accept_transfer) > 0:
                bus_received_in = feasible_buses_to_accept_transfer[np.random.randint(low=0, high=len(feasible_buses_to_accept_transfer))]

                try:
                    self.move_student_and_update_memo(student_name, bus_sent_from, bus_received_in)

                    if len(self.state[bus_sent_from]) == 1:
                        self.buses_not_filled_minimally.remove(bus_sent_from)
                        self.buses_filled_minimally.append(bus_sent_from)
                    elif bus_sent_from not in self.buses_not_filled_minimally and bus_sent_from in self.buses_filled_minimally:
                        self.buses_not_filled_minimally.append(bus_sent_from)
                        self.buses_filled_minimally.remove(bus_sent_from)

                    if len(self.state[bus_received_in]) == self.size_bus:
                        self.buses_not_filled_maximally.remove(bus_received_in)
                        self.buses_filled_maximally.append(bus_received_in)
                    elif bus_received_in not in self.buses_not_filled_maximally and bus_received_in in self.buses_filled_maximally:
                        self.buses_not_filled_maximally.append(bus_received_in)
                        self.buses_filled_maximally.remove(bus_received_in)
                except:
                    pass

    def swap(self):
        student_names = ['', '']
        buses = [-1, -1]
        num_attempts = 10 # infinite loop catch in case all buses are minimally filled
        while (student_names[0] == '' or student_names[1] == '') and num_attempts > 0:
            student_names[0] = self.student_names[np.random.sample(self.student_names)]
            student_names[1] = self.student_names[np.random.sample(self.student_names)]
            buses = list(map(lambda student_name: self.student_assignments[self.name_to_index[student_name]]))
            if student_names[0] == student_names[1] or buses[0] == buses[1]:
                student_names = ['', '']
            num_attempts -= 1

        if student_names[0] == '' or student_names[1] == '':
            return None

        try:
            if num_attempts >= 0:
                for i in range(2):
                    bus_sent_from = buses[i]
                    bus_received_in = buses[1 - i]
                    self.move_student_and_update_memo(student_names[i], bus_sent_from, bus_received_in)
        except:
            pass

    def permutation(self):
        num_students_in_cyle = int(np.random.exponential(scale=2)) + 3
        student_names = map(lambda index: self.student_names[index], np.random.choice(len(self.student_names), num_students_in_cyle))
        buses = list(map(lambda student_name: self.student_assignments[self.name_to_index[student_name]], student_names))
        permutation = np.random.permutation(num_students_in_cyle)

        for i in range(num_students_in_cyle):
            bus_sent_from = buses[i]
            bus_received_in = buses[permutation[i]]
            try:
                self.move_student_and_update_memo(student_names[i], bus_sent_from, bus_received_in)
            except:
                pass

    def move_student_and_update_memo(self, student_name, bus_sent_from, bus_received_in):
        student_index = -1
        student_index = self.name_to_index[student_name]
        # move student
        try:
            self.state[bus_sent_from].remove(str(student_index))
            self.state[bus_received_in].append(str(student_index))
            self.student_assignments[student_index] = bus_received_in
            # update memo
            rowdy_groups_student_is_in = self.rowdy_group_student_membership_matrix[:, student_index]

            # update bus student is sent from
            self.fraction_of_rowdy_group_in_bus[bus_sent_from,:] -= np.multiply(self.scaled_rowdy_group_student_membership_matrix[:, student_index], rowdy_groups_student_is_in)
            friendships_in_bus_sent_from_for_student = self.friendships_in_bus_for_student[student_index][bus_sent_from]
            for friend_name in self.graph.adj[student_name]:
                friend_index = self.name_to_index[friend_name]
                self.friendships_in_bus_for_student[friend_index][bus_sent_from].remove(student_index)
            delta_number_of_friendships_in_bus_sent_from_for_rowdy_group = np.zeros(self.rowdy_group_student_membership_matrix.shape[0])
            rowdy_groups_of_friends_in_bus_sent_from = list(map(lambda friend_index: np.where(self.rowdy_group_student_membership_matrix[:,friend_index] == 1), friendships_in_bus_sent_from_for_student))
            for i in range(len(rowdy_groups_of_friends_in_bus_sent_from)):
                rowdy_group_indices = rowdy_groups_of_friends_in_bus_sent_from[i]
                for rowdy_group_index in rowdy_group_indices:
                    delta_number_of_friendships_in_bus_sent_from_for_rowdy_group[rowdy_group_index] += 1
            self.number_of_friendships_in_bus_for_rowdy_group[bus_sent_from,:] -= delta_number_of_friendships_in_bus_sent_from_for_rowdy_group

            # update bus student is received in
            self.fraction_of_rowdy_group_in_bus[bus_received_in,:] += np.multiply(self.scaled_rowdy_group_student_membership_matrix[:, student_index], rowdy_groups_student_is_in)
            friendships_in_bus_received_in_for_student = self.friendships_in_bus_for_student[student_index][bus_received_in]
            for friend_name in self.graph.adj[student_name]:
                friend_index = self.name_to_index[friend_name]
                self.friendships_in_bus_for_student[friend_index][bus_received_in].append(student_index)
            delta_number_of_friendships_in_bus_received_in_for_rowdy_group = np.zeros(self.rowdy_group_student_membership_matrix.shape[0])
            rowdy_groups_of_friends_in_bus_received_in = list(map(lambda friend_index: np.where(self.rowdy_group_student_membership_matrix[:,friend_index] == 1), friendships_in_bus_received_in_for_student))
            for i in range(len(rowdy_groups_of_friends_in_bus_received_in)):
                rowdy_group_indices = rowdy_groups_of_friends_in_bus_received_in[i]
                for rowdy_group_index in rowdy_group_indices:
                    delta_number_of_friendships_in_bus_received_in_for_rowdy_group[rowdy_group_index] += 1
            self.number_of_friendships_in_bus_for_rowdy_group[bus_received_in,:] += delta_number_of_friendships_in_bus_received_in_for_rowdy_group
        except:
            1 / 0

    def energy(self):
        """Calculates the length of the route."""
        # nullify all friendships of students for which
        # at least one of the students belongs to a rowdy group
        # which is fully present in the students' bus
        count = 0
        number_friendships_per_student = np.zeros(len(self.student_names))
        removed_rowdy_groups = np.where(self.fraction_of_rowdy_group_in_bus == 1)[1].tolist()
        removed_students = []
        for rowdy_group_index in removed_rowdy_groups:
            removed_students += self.constraints[rowdy_group_index]
        for student_name in self.student_names:
            if student_name not in removed_students:
                student_index = self.name_to_index[student_name]
                number_friendships_per_student[student_index] = len(self.friendships_in_bus_for_student[student_index])
        # sum up all remaining friendships
        return self.max_energy - np.sum(number_friendships_per_student)

def dict_to_string(dict):
    result = ""
    for key in dict:
        if len(dict[key]) != 0:
            result = result + str(dict[key]) + "\n"
        else:
            result = result + "[]" + "\n"
    return result


def solve(graph, num_buses, size_bus, constraints):
    # TODO: Write this method as you like. We'd recommend changing the arguments here as well
    # graph, num_buses, size_bus, constraints = parse_input(path_to_outputs)

    # construct a dictionary mapping each person to the rowdy groups they're in

    student_names = list(graph.nodes())
    name_to_index = dict()

    for i, student in enumerate(student_names):
        name_to_index[student] = i

    num_rowdy_groups = len(constraints)
    bus_assignments = {}
    for i in range(num_buses):
        bus_assignments[i] = []
    # M
    rowdy_group_student_membership_matrix = np.zeros((len(constraints), len(student_names)))
    for rowdy_group_index in range(len(constraints)):
        for student_name in student_names:
            if student_name in constraints[rowdy_group_index]:
                rowdy_group_student_membership_matrix[rowdy_group_index, name_to_index[student_name]] = 1

    sums = np.sum(rowdy_group_student_membership_matrix, axis=1)

    scaled_rowdy_group_student_membership_matrix = rowdy_group_student_membership_matrix / sums[:, None]
    # C
    fraction_of_rowdy_group_in_bus = np.zeros(shape=(num_buses, num_rowdy_groups), dtype=float)
    # L
    number_of_friendships_in_bus_for_rowdy_group = np.zeros(shape=(num_buses, num_rowdy_groups), dtype=float)
    # sort students by number of rowdy groups they're in
    # iterate through every person
    student_ordering = np.argsort(-np.sum(rowdy_group_student_membership_matrix, axis=0))
    for i, student_index in enumerate(student_ordering[:num_buses]):
        update_data(student_index, i, student_names, rowdy_group_student_membership_matrix,
                    np.zeros(shape=(num_buses, num_rowdy_groups)), fraction_of_rowdy_group_in_bus,
                    number_of_friendships_in_bus_for_rowdy_group,
                    bus_assignments, scaled_rowdy_group_student_membership_matrix)

    for student_index in student_ordering[num_buses:]:
        additional_friendships = np.zeros(shape=(num_buses, 1), dtype=float)
        friend_count_in_rgs = np.zeros(shape=(num_buses, num_rowdy_groups))

        for bus in range(num_buses):
            count = 0
            for friend in graph.adj[student_names[student_index]]:
                if name_to_index[friend] in bus_assignments[bus]:
                    friend_rg = rowdy_group_student_membership_matrix[:, name_to_index[friend]]
                    student_rg = rowdy_group_student_membership_matrix[:, student_index]
                    common_rgs = np.where(friend_rg + student_rg == 2)
                    for common_rg in common_rgs:
                        friend_count_in_rgs[bus, common_rg] += 1
                    count += 1
            additional_friendships[bus] = count

        number_of_friendships_in_bus_for_rowdy_group_temp = number_of_friendships_in_bus_for_rowdy_group + additional_friendships @ rowdy_group_student_membership_matrix[:,student_index].reshape(1, num_rowdy_groups)
        fraction_of_rowdy_group_in_bus_temp = fraction_of_rowdy_group_in_bus + np.ones(shape=additional_friendships.shape) @ scaled_rowdy_group_student_membership_matrix[:, student_index].reshape(1,num_rowdy_groups)
        floored_fraction_of_rowdy_group_in_bus_temp = np.floor(fraction_of_rowdy_group_in_bus_temp)

        reward_vector = additional_friendships.reshape(additional_friendships.size)
        cost_matrix = np.multiply(floored_fraction_of_rowdy_group_in_bus_temp,
                                  number_of_friendships_in_bus_for_rowdy_group_temp) + fraction_of_rowdy_group_in_bus_temp / num_rowdy_groups
        rowdy_groups_student_is_in = rowdy_group_student_membership_matrix[:, student_index]
        cost_vector = cost_matrix @ rowdy_groups_student_is_in
        heuristics = reward_vector - cost_vector
        sorted_heuristic_indices = np.argsort(-heuristics)

        for index in sorted_heuristic_indices:
            load = len(bus_assignments[index])
            if load < size_bus:
                update_data(student_index, index, student_names, rowdy_group_student_membership_matrix, friend_count_in_rgs,
                            fraction_of_rowdy_group_in_bus, number_of_friendships_in_bus_for_rowdy_group,
                            bus_assignments, scaled_rowdy_group_student_membership_matrix)
                break

    # create a matrix mapping a student's number of friends in the bus they're in
    friendships_in_bus_for_student = [[[] for j in range(num_buses)] for i in range(len(student_names))] # np.zeros(shape=(num_buses, len(student_names)))
    for student in student_names:
        for friend in graph.adj[student]:
            for bus in bus_assignments:
                if friend in bus_assignments[bus]:
                    friendships_in_bus_for_student[name_to_index[student]][bus].append(name_to_index[friend])

    tsp = SimulatedAnnealer(bus_assignments, friendships_in_bus_for_student, fraction_of_rowdy_group_in_bus, rowdy_group_student_membership_matrix,
                            name_to_index, student_names,
                            number_of_friendships_in_bus_for_rowdy_group,
                            scaled_rowdy_group_student_membership_matrix, constraints, size_bus, graph)

    tsp.steps = 10000
    # since our state is just a list, slice is the fastest way to copy
    # tsp.copy_strategy = "slice"
    if tsp.energy() > 0:
        state, e = tsp.anneal()
        return dict_to_string(state)
    else:
        return dict_to_string(tsp.state)


# Update memoized data
def update_data(student_index, bus_index, student_names, rowdy_group_student_membership_matrix, friend_count_in_rgs, fraction_of_rowdy_group_in_bus, number_of_friendships_in_bus_for_rowdy_group, bus_assignments, scaled_rowdy_group_student_membership_matrix):
    bus_assignments[bus_index].append(student_names[student_index])
    rowdy_groups_student_is_in = rowdy_group_student_membership_matrix[:, student_index]
    fraction_of_rowdy_group_in_bus[bus_index,:] += np.multiply(scaled_rowdy_group_student_membership_matrix[:, student_index], rowdy_groups_student_is_in)
    number_of_friendships_in_bus_for_rowdy_group += friend_count_in_rgs


def decrease(student_index, bus_index, student_names, rowdy_group_student_membership_matrix, friend_count_in_rgs, fraction_of_rowdy_group_in_bus, number_of_friendships_in_bus_for_rowdy_group, bus_assignments, scaled_rowdy_group_student_membership_matrix):
    bus_assignments[bus_index].remove(student_names[student_index])
    rowdy_groups_student_is_in = rowdy_group_student_membership_matrix[:, student_index]
    fraction_of_rowdy_group_in_bus[bus_index,:] -= np.multiply(scaled_rowdy_group_student_membership_matrix[:, student_index], rowdy_groups_student_is_in)
    number_of_friendships_in_bus_for_rowdy_group -= friend_count_in_rgs


def main():
    '''
        Main method which iterates over all inputs and calls `solve` on each.
        The student should modify `solve` to return their solution and modify
        the portion which writes it to a file to make sure their output is
        formatted correctly.
    '''
    size_categories = ["small", "medium", "large"]
    if not os.path.isdir(path_to_outputs):
        os.mkdir(path_to_outputs)

    for size in size_categories:
        category_path = path_to_inputs + "/" + size
        output_category_path = path_to_outputs + "/" + size
        category_dir = os.fsencode(category_path)

        if not os.path.isdir(output_category_path):
            os.mkdir(output_category_path)

        for input_folder in os.listdir(category_dir):
            input_name = os.fsdecode(input_folder)
            graph, num_buses, size_bus, constraints = '', '', '', ''
            execute = True
            try:
                graph, num_buses, size_bus, constraints = parse_input(category_path + "/" + input_name)
                graph.remove_edges_from(graph.selfloop_edges())
            except:
                execute = False
            if execute:
                solution = solve(graph, num_buses, size_bus, constraints)
                output_file = open(output_category_path + "/" + input_name + ".out", "w")

                # TODO: modify this to write your solution to your
                #      file properly as it might not be correct to
                #      just write the variable solution to a file
                output_file.write(solution)

                output_file.close()

if __name__ == '__main__':
    main()
