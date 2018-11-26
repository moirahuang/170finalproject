# 170project

Our current idea for approaching the project is to iteratively assign each person to a bus, each time doing the best possible assignment. To find this assignment, we'll use ILP for each individual person. There will be indicators for friendships, and for each rowdy group, you create a unique variable and assign each person in the rowdy group the weight of 1/size of the group. As long as no rowdy group is fulfilled, the constraint that the sum of the weights being less than 1 will always be satisfied. 
