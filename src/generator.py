import numpy as np
import random
from solver import is_valid

def generate_sudoku(difficulty=0.5):
    # 1. Önce boş bir tahta oluştur
    board = np.zeros((9, 9), dtype=int)
    
    # 2. Tahtayı tam olarak doldurur (geçerli bir tam çözüm üretir)
    def fill_board(b):
        for r in range(9):
            for c in range(9):
                if b[r][c] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)
                    for n in nums:
                        if is_valid(b, n, (r, c)):
                            b[r][c] = n
                            if fill_board(b): return True
                            b[r][c] = 0
                    return False
        return True

    fill_board(board)
    
    # 3. Zorluk seviyesine göre rakamları rastgele ayarlar
    # difficulty 0.1 (çok kolay) ile 0.8 (çok zor) arası olabilir
    attempts = int(81 * difficulty)
    while attempts > 0:
        row = random.randint(0, 8)
        col = random.randint(0, 8)
        if board[row][col] != 0:
            board[row][col] = 0
            attempts -= 1
            
    return board