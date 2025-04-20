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
    for digit1 in puzzle.value_set:
        for digit2 in puzzle.value_set:
            for digit3 in puzzle.value_set:
                for digit4 in puzzle.value_set:
                    my_attempt.variable_values = [digit1, digit2, digit3, digit4]
                    try:
                        result = puzzle.evaluate(my_attempt.variable_values)
                        if result == 1:
                            return my_attempt.variable_values
                    except:
                        pass

    # <==== insert your code above here
    
    # should never get here
    return [-1, -1, -1, -1]

def get_names(namearray: np.ndarray) -> list:
    family_names = []
    # ====> insert your code below here

    for i in range(namearray.shape[0]):
        surname_chars = namearray[i, -6:]
        family_names.append("".join(surname_chars))
    
    # <==== insert your code above here
    return family_names

def check_sudoku_array(attempt: np.ndarray) -> int:
    tests_passed = 0
    slices = []  # this will be a list of numpy arrays
    
    # ====> insert your code below here


    # use assertions to check that the array has 2 dimensions each of size 9
    assert attempt.shape == (9, 9), "Array must be 9x9"
    
    ## Remember all the examples of indexing above
    ## and use the append() method to add something to a list
    for i in range(9):
        row = attempt[i, :]
        slices.append(row)

    for j in range(9):
        col = attempt[:,j]
        slices.append(col)
    
    for i in range(0,9,3):
        for j in range(0,9,3):
            box = attempt[i:i+3, j:j+3]
            slices.append(box.flatten())

    

    for slice in slices:  # easiest way to iterate over list
        if len(np.unique(slice)) == 9:  # get number of unique values in slice
            tests_passed += 1  # increment value of tests_passed as appropriate
        
    # <==== insert your code above here
    # return count of tests passed
    return tests_passed
