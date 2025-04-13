from approvedimports import *

def exhaustive_search_4tumblers(puzzle: CombinationProblem) -> list:
    """simple brute-force search method that tries every combination until
    it finds the answer to a 4-digit combination lock puzzle.
    """

    # check that the lock has the expected number of digits
    assert puzzle.numdecisions == 4, "this code only works for 4 digits"

    # create an empty candidate solution
    my_attempt = CandidateSolution()
    
    # ====> insert your code below here

     for i in range(10):  
        for j in range(10):  
            for k in range(10):  
                for l in range(10):  
                  
                    my_attempt.variable_values = [i, j, k, l]
                    
               
                    score = puzzle.evaluate(my_attempt.variable_values)

                    if score == 4:
                        return [i, j, k, l]
    

    # <==== insert your code above here
    
    # should never get here
    return [-1, -1, -1, -1]

def get_names(namearray: np.ndarray) -> list:
    family_names = []
    # ====> insert your code below here
     for name in namearray:
        family_names.append(name[-6:])
        
    
    # <==== insert your code above here
    return family_names

def check_sudoku_array(attempt: np.ndarray) -> int:
    tests_passed = 0
    slices = []  # this will be a list of numpy arrays
    
    # ====> insert your code below here


    # use assertions to check that the array has 2 dimensions each of size 9
    assert attempt.ndim == 2, "Array must have 2 dimensions"
    assert attempt.shape == (9, 9), "Array must be of size 9x9"
    
     for row in attempt:
        slices.append(row)

     for col in attempt.T:  
        slices.append(col)

        for i in range(0, 9, 3):  
        for j in range(0, 9, 3):
            subgrid = attempt[i:i+3, j:j+3].flatten() 
            slices.append(subgrid)


    ## Remember all the examples of indexing above
    ## and use the append() method to add something to a list

    for slice in slices:  # easiest way to iterate over list
        pass
        # print(slice) - useful for debugging?

        # get number of unique values in slice
          unique_values = np.unique(slice)

        # increment value of tests_passed as appropriate
        if len(unique_values) == 9:
            tests_passed += 1
    
    # <==== insert your code above here
    # return count of tests passed
    return tests_passed
