import numpy as np

def is_valid(board, num, pos):
    r, c = pos
    if num in board[r]: return False
    if num in board[:, c]: return False
    bx, by = c // 3, r // 3
    for i in range(by*3, by*3+3):
        for j in range(bx*3, bx*3+3):
            if board[i][j] == num: return False
    return True

def solve_logic(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for num in range(1, 10):
                    if is_valid(board, num, (r, c)):
                        board[r][c] = num
                        if solve_logic(board): return True
                        board[r][c] = 0
                return False
    return True