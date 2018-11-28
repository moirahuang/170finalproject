import networkx as nx
import cvxpy
import numpy as np
import os

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

def solve():
    #TODO: Write this method as you like. We'd recommend changing the arguments here as well
    graph, num_buses, size_bus, constraints = parse_input(path_to_outputs)

    # construct a dictionary mapping each person to the rowdy groups they're in
    students = list(graph.nodes)
    num_rowdy_groups = len(constraints)
    bus_assignments = {}
    for i in range(num_buses):
        bus_assigments[i] = []
    #M
    rowdy_group_to_students = np.zeros((num_rowdy_groups, len(students)))
    for rowdy_group_index in range(num_rowdy_groups):
        for student_index in range(len(students)):
            if constraints[rowdy_group_index].contains(students[student_index]):
                rowdy_group_to_students[rowdy_group_index, student_index] = 1

    sums = np.sum(rowdy_group_to_students, axis=1)
    scaled_rowdy_group_to_students = rowdy_group_to_students / sums[:,None]
    #C
    fraction_of_rowdy_group_in_bus = np.zeros(shape=(num_buses, num_rowdy_groups))
    #L
    number_of_friendships_in_bus_for_rowdy_group = np.zeros(shape=(num_buses, num_rowdy_groups))
    #I
    floored_fraction_of_rowdy_group_in_bus = np.zeros(shape=(num_buses, num_rowdy_groups))

    #sort students by number of rowdy groups they're in
    # iterate through every person
    for student in np.argsort(np.sum(rowdy_group_to_students, axis=0)):
        additional_friendships = np.zeros(shape=(num_buses, 1)))
        for bus in range(num_buses):
            count = 0
            for friend in graph.adj(student):
                if np.any(bus[:, 0] == friend):
                    count+=1
            additional_friendships[bus] = count

        number_of_friendships_in_bus_for_rowdy_group_temp = number_of_friendships_in_bus_for_rowdy_group + rowdy_group_to_students[:,student] @ additional_friendships.T
        fraction_of_rowdy_group_in_bus_temp               = fraction_of_rowdy_group_in_bus + scaled_rowdy_group_to_students[:,studnet] @ np.ones(shape=additional_friendships.shape).T
        floored_fraction_of_rowdy_group_in_bus_temp       = np.floor(fraction_of_rowdy_group_in_bus_temp)

        heuristic = additional_friendships - number_of_friendships_in_bus_for_rowdy_group_temp.T @ (floored_fraction_of_rowdy_group_in_bus_temp + fraction_of_rowdy_group_in_bus_temp / num_rowdy_groups)
        


        #append friend to bus

# Solving the problem
knapsack_problem.solve(solver=cvxpy.GLPK_MI)
    # heuristic is # of friendships created - factor * potential rowdy groups created (but make -large value if a rowdy group is created--recalculate friendships if a rowdy group has to be made)

    # simulated anealing to swap for best solution

    pass

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
            graph, num_buses, size_bus, constraints = parse_input(category_path + "/" + input_name)
            solution = solve(graph, num_buses, size_bus, constraints)
            output_file = open(output_category_path + "/" + input_name + ".out", "w")

            #TODO: modify this to write your solution to your
            #      file properly as it might not be correct to
            #      just write the variable solution to a file
            output_file.write(solution)

            output_file.close()

if __name__ == '__main__':
    main()
