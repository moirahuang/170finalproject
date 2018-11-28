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
    rowdy_group_to_students = np.zeros((len(constraints), len(list(students))))
    for rowdy_group_index in range(len(constraints)):
        for student_index in range(len(students)):
            if constraints[rowdy_group_index].contains(students[student_index]):
                rowdy_group_to_students[rowdy_group_index, student_index] = 1


    buses = np.zeros(shape=(num_buses, size_bus))
    # iterate through every person
    for student in students:
        # make list of friendships in each bus
        friends = np.zeros(shape=(num_buses, 1))
        # iterate through the buses and construct an array with the number of friends they have on the buses
        for bus in buses:
            count = 0
            for friend in graph.adj(student):
                if np.any(bus[:, 0] == friend):
                    count+=1
            friends[]

        friends = np.array(friends)
        # ILP to pick the bus with the max heuristic (based on vectors)
        P = size_bus
        weights = friends
        utilities = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72]) #rowdy groups

        # The variable we are solving for
        selection = cvxpy.Bool(len(buses))

        # The sum of the weights should be less than or equal to P
        weight_constraint = buses + selection <= P

        # Our total utility is the sum of the item utilities
        total_utility = utilities * selection

        # We tell cvxpy that we want to maximize total utility
        # subject to weight_constraint. All constraints in
        # cvxpy must be passed as a list
        knapsack_problem = cvxpy.Problem(cvxpy.Maximize(total_utility), [weight_constraint])

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
