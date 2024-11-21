import math
from os import write
from queue import PriorityQueue
import random
from re import T
import numpy as np

def available_direction(x, n):  # this function will generate a list of possible moves
    moves = ['L', 'R', 'U', 'D']
    if x % n == 0:  # the blank cell is at column 0 so it cannot be moved any further to the left
        moves.remove('L')
    if x % n == n - 1:  # the blank cell is at column n - 1 so it cannot be moved any further to the right
        moves.remove('R')
    if x // n == 0:  # the blank cell is at row 0 so it cannot be moved up
        moves.remove('U')
    if x // n == n - 1:  # the blank cell is at row n - 1 so it cannot be moved down
        moves.remove('D')
    return moves

def puzzle_generator(n, random_step = 50):
    puzzle = list([i for i in range(1, n * n)])
    puzzle.append(0)  # we start at the solution configuration
    for i in range(random_step):  # we simulate a number of steps -- 50 steps by default
        zero_index = puzzle.index(0)  # find the index of the blank cell
        available_moves = available_direction(zero_index, n)  # check its vailable moves
        move = random.choice(available_moves) # choose a random move
        if move == 'L':  # execute it
            puzzle[zero_index], puzzle[zero_index - 1] = puzzle[zero_index - 1], puzzle[zero_index]
        elif move == 'R':
            puzzle[zero_index], puzzle[zero_index + 1] = puzzle[zero_index + 1], puzzle[zero_index]
        elif move == 'U':
            puzzle[zero_index], puzzle[zero_index - n] = puzzle[zero_index - n], puzzle[zero_index]
        elif move == 'D':
            puzzle[zero_index], puzzle[zero_index + n] = puzzle[zero_index + n], puzzle[zero_index]
    return np.array(puzzle).reshape(n, n)  # return the board after making random_step moves away from the solution configuration

def heuristic(state, solution, n):  # Manhattan heuristic -- Feel free to come up with your own heuristic (bearing in mind that if the heuristic is not admissible, A* will not always find the optimal plan)
    res = 0
    for i in range(n**2-1):
      num = state[i]
      sorted_pos = solution[num-1]
      res += abs(i - sorted_pos)
    return res  # return the total Manhanttan distance over all number

def direction(parent_state, child_state, n): # find the direction (i.e., U, D, L, or R) in which we can move from the parent to its immediate child
    res = ''
    parent_zero = parent_state.index(0)
    child_zero = child_state.index(0)
    if child_zero > parent_zero:
      if child_zero-1 == parent_zero:
        res = 'R'
      else:
        res = 'D'
    else:
      if child_zero+1 == parent_zero:
        res = 'L'
      else:
        res = 'U'
    return res

def trace(state, parent, n):
    res = ''
    while parent[state] != None:  # if we are still at an intermediate state
        res = direction(parent[state], state, n) + res  # find the direction that moves parent[state] to state and add it to the plan
        state = parent[state]  # trace back to the parent
    return res  # return the plan

def generate(state, g, solution, n):
    res = []  # the list of children nodes

    zero_index = state.index(0)
    available_moves = available_direction(zero_index, n)  # find the list of possible move in the current state

    for move in available_moves: # for each move, we move to a different next state
      if move == 'L':
        new_state = tuple(state[:zero_index-1] + (state[zero_index],) + (state[zero_index-1],) + state[zero_index+1:])
        new_h = heuristic(new_state, solution, n)
        new_g = g+1
        res.append((new_h,new_g,new_state))
      elif move == 'R':
        new_state = tuple(state[:zero_index] + (state[zero_index+1],) + (state[zero_index],) + state[zero_index+2:])
        new_h = heuristic(new_state, solution, n)
        new_g = g+1
        res.append((new_h,new_g,new_state))
      elif move == 'U':
        new_state = tuple(state[:zero_index-n] + (state[zero_index],) + state[(zero_index-n+1):zero_index] + (state[zero_index-n],) + state[zero_index+1:])
        new_h = heuristic(new_state, solution, n)
        new_g = g+1
        res.append((new_h,new_g,new_state))
      elif move == 'D':
        new_state = tuple(state[:zero_index] + (state[zero_index+n],) + state[(zero_index+1):(zero_index+n)] + (state[zero_index],) + state[zero_index+n+1:])
        new_h = heuristic(new_state, solution, n)
        new_g = g+1
        res.append((new_h,new_g,new_state))
        # FILL IN THE REST OF THE CODE FOR THIS FOR LOOP TO GENERATE ALL CHILDREN NODES
        # MAKE SURE YOU READ THE REST OF THE TEMPLATE OF THE SEARCH FUNCTION TO UNDERSTAND THAT WE REPRESENT A NODE = (H-VALUE, G-VALUE, NEXT STATE)

    return res  # return the list of NODES

def search(puzzle, n):  # this is the main function definition which CANNOT be changed, i.e., its name and expected input/output must NOT be changed -- otherwise, the content of the function can be changed as you see fit
    frontier = PriorityQueue()  # set up the frontier as a priority queue
    explored = dict([])  # set up the explored table, which is now a dictionary mapping from a state to its current f value
    parent = dict([])  # we also set up the parent table, which is a dictionary mapping from a state Y to its parent state X according to the current best plan to reach X

    state = tuple(puzzle.reshape(1, n * n)[0].tolist())  # flatten the N x N puzzle table into a flat vector
    parent[state] = None  # the initial state has no parent

    solution = list([i for i in range(1, n * n)])  # set up the solution state
    solution.append(0)
    solution = tuple(solution)  # why do we use tuple? tuple is hashable while list is not -- so if we want to use this as key to a dictionary, it must be in tuple form

    h, g = heuristic(state, solution, n), 0  # compute the heuristic (h) and actual-cost-so-far (g) values for the initial state
    node = (h, g, state)  # create the initial node which is a triplet (heuristic estimate to solution, actual cost so far, state)
    # MAKE SURE YOU DISTINGUISH BETWEEN PROBLEM STATE & SEARCH NODE IN THE REST OF THE IMPLEMENTATION

    frontier.put((h + g, node)) # put the node on a priority queue: add its corresponding (f, node)-tuple to the priority queue so that the node is prioritized by its f = h + g
    i = 0
    while not frontier.empty():  # while the frontier is not empty
        (f, (h, g, state)) = frontier.get()  # unpack the value of the (f, node)-tuple with top priority -- (f, node) where node = (h, g, state)
        #i += 1
        #print(f"{state},{i}")
        
        if state in explored:  # if it has been explored before, this node corresponds to a sub-optimal plan
            continue  # let's ignore it and move on

        explored[state] = f  # otherwise, record that state is now explored and the current best plan to reach it has value f
        if state == solution:  # if it is a solution
            return trace(state, parent, n)  # run the tracing procedure to produce the corresponding set of moves

        children = generate(state, g, solution, n)  # otherwise, generate a set of its children nodes
        for child in children: # for each child
           inFrontier = False
           frontierUpdated = False
           
           if (child[2] in explored.keys() and explored[child[2]] <= child[0] + child[1]):
              continue
           
           elif child[2] not in explored.keys():
              for (f2,node2) in frontier.queue:
                 if node2[2] == child[2]:
                    inFrontier = True
                    if f2 > (child[0] + child[1]):
                       frontier.queue.remove((f2,node2))
                       if frontierUpdated == False:
                          frontier.put((child[0] + child[1], child))
                          parent[child[2]] = state
                          frontierUpdated = True
                    else:
                        break
              if(inFrontier == False):
                 frontier.put((child[0] + child[1], child))
                 parent[child[2]] = state

           elif (child[2] in explored.keys() and explored[child[2]] > child[0] + child[1]):
              for (f2,node2) in frontier.queue:
                 if node2[2] == child[2]:
                    if f2 > child[0] + child[1]:
                       frontier.queue.remove((f2,node2))
                       if frontierUpdated == False:
                          frontier.put((child[0] + child[1], child))
                          parent[child[2]] = state
                          explored[child[2]] = child[0] + child[1]
                          frontierUpdated = True
                    else:
                       break
                    inFrontier = True
              if inFrontier == False:
                 frontier.put((child[0] + child[1], child))
                 parent[child[2]] = state

           else:
              for (f2,node2) in frontier.queue:
                 if node2[2] == child[2]:
                    if f2 > child[0] + child[1]:
                       frontier.queue.remove((f2,node2))
                       if frontierUpdated == False:
                          frontier.put((child[0] + child[1], child))
                          parent[child[2]] = state
                          frontierUpdated = True
                
    return None

while (True):
    n = int(input("n:"))  # set the board size
    puzzle = puzzle_generator(n)  # generate a test instance (with the default 50 steps of random moves)

    print(puzzle)  # print out the initial state
    plan = search(puzzle, n)
    if(plan):
        print(plan)  # print out the plan
        print(len(plan))  # print out the length of the plan -- this should be the minimum number of steps to reach the solution if the heuristic is admissible
    else:
        flatPuzzle = tuple(puzzle.reshape(1, n * n)[0].tolist())# flatten the N x N puzzle table into a flat vector
        solution = list([i for i in range(1, n * n)])  # set up the solution state
        solution.append(0)
        solution = tuple(solution)
        if (flatPuzzle == solution):
            print("The puzzle is already solved!")
        else:
            print("No solution found!")
