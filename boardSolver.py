#!/usr/bin/python
# going to take a more functional programming approach
# cross taken from 'norvig' (clever idea) 

def cross(A,B):
    return [a+b for a in A for b in B]

def initBoard():
    digits  = '123456789'
    letters = 'ABCDEFGHI'
    squares = cross(letters,digits)
    # we need to check for each square the row, column, and box
    # which values can viably be used for the solution
    # maintain a dict for each square of the associated row, column, box vals
    # (this is for forward checking)
    rows = [cross(let,digits) for let in letters]
    cols = [cross(letters,dig) for dig in digits]
    boxes = [cross(letters[i:i+3], digits[j:j+3]) for j in range(0,9,3) for i in range(0,9,3)]
    print boxes
if __name__ == "__main__":
    initBoard()

