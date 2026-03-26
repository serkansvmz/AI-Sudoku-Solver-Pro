import tkinter as tk
from tkinter import filedialog, messagebox, ttk 
import numpy as np
import torch
import time  
from model import SudokuCNN
from solver import is_valid, solve_logic
from utils import predict_digit, extract_sudoku_warp
from generator import generate_sudoku

# --- Model ve Cihaz Hazırlığı ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuCNN().to(device)
try:
    model.load_state_dict(torch.load("sudoku_cnn.pth", map_location=device))
    model.eval()
except Exception as e:
    print(f"Model yüklenemedi: {e}")

root = tk.Tk()
root.title("AI Sudoku Solver & Game")

entries = []
board = np.zeros((9, 9), dtype=int)

# --- ZAMANLAYICI DEĞİŞKENLERİ ---
start_time = None
running = False

def update_timer():
    if running:
        total_seconds = int(time.time() - start_time)
        minutes, seconds = divmod(total_seconds, 60)
        timer_label.config(text=f"Süre: {minutes:02d}:{seconds:02d}")
        root.after(1000, update_timer)

# --- OYUN KONTROLLERİ ---

def check_win():
    global running
    for r in range(9):
        for c in range(9):
            val = entries[r][c].get()
            if not val.isdigit(): return False
            num = int(val)
            temp = board[r][c]
            board[r][c] = 0
            if not is_valid(board, num, (r, c)):
                board[r][c] = temp
                return False
            board[r][c] = num
    
    running = False # Kazanınca süreyi durdur
    return True

def validate_entry(r, c):
    val = entries[r][c].get()
    if val == "":
        default_bg = "#e0e0e0" if (r // 3 + c // 3) % 2 == 0 else "white"
        entries[r][c].config(bg=default_bg)
        board[r][c] = 0
        return

    if val.isdigit() and 1 <= int(val) <= 9:
        num = int(val)
        temp_board = board.copy()
        temp_board[r][c] = 0
        
        if is_valid(temp_board, num, (r, c)):
            entries[r][c].config(bg="#c8e6c9")
            board[r][c] = num
            if check_win():
                messagebox.showinfo("TEBRİKLER", f"Kazandınız!\n{timer_label.cget('text')}")
        else:
            entries[r][c].config(bg="#ffcdd2")
            board[r][c] = 0
    else:
        entries[r][c].delete(0, tk.END)

# --- BUTON FONKSİYONLARI ---

def start_detect():
    path = filedialog.askopenfilename()
    if not path: return
    warp = extract_sudoku_warp(path)
    if warp is None:
        messagebox.showerror("Hata", "Sudoku ızgarası bulunamadı!")
        return
    
    for r in range(9):
        for c in range(9):
            cell_s = 450 // 9
            cell = warp[r*cell_s:(r+1)*cell_s, c*cell_s:(c+1)*cell_s]
            digit = predict_digit(cell, model, device)
            entries[r][c].delete(0, tk.END)
            if digit != 0:
                entries[r][c].insert(0, str(digit))
                validate_entry(r, c)

def start_new_game():
    global start_time, running
    
    # Zorluk seviyesini menüden ayarlama
    diff_name = difficulty_var.get()
    diff_map = {"Kolay": 0.3, "Orta": 0.5, "Zor": 0.7}
    difficulty = diff_map.get(diff_name, 0.5)

    for r in range(9):
        for c in range(9):
            entries[r][c].delete(0, tk.END)
            bg = "#e0e0e0" if (r // 3 + c // 3) % 2 == 0 else "white"
            entries[r][c].config(bg=bg)
            board[r][c] = 0
    
    new_puzzle = generate_sudoku(difficulty)
    for r in range(9):
        for c in range(9):
            if new_puzzle[r][c] != 0:
                entries[r][c].insert(0, str(new_puzzle[r][c]))
                validate_entry(r, c)
    
    # Zamanlayıcıyı başlatma
    start_time = time.time()
    running = True
    update_timer()

def animated_solve():
    global running
    running = False # AI çözerken süreyi durdurur
    for r in range(9):
        for c in range(9):
            val = entries[r][c].get()
            board[r][c] = int(val) if val.isdigit() else 0

    def solve_step():
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    for num in range(1, 10):
                        if is_valid(board, num, (r, c)):
                            board[r][c] = num
                            entries[r][c].delete(0, tk.END)
                            entries[r][c].insert(0, str(num))
                            entries[r][c].config(fg="blue", bg="#c8e6c9")
                            root.update()
                            if solve_step(): return True
                            board[r][c] = 0
                            entries[r][c].delete(0, tk.END)
                    return False
        return True
    solve_step()

# --- ARAYÜZ KURULUMU ---

def create_grid():
    for i in range(9):
        row = []
        for j in range(9):
            bg = "#e0e0e0" if (i // 3 + j // 3) % 2 == 0 else "white"
            e = tk.Entry(root, width=2, font=('Arial', 18, 'bold'), justify='center', bg=bg)
            e.grid(row=i, column=j, padx=1, pady=1)
            e.bind('<KeyRelease>', lambda event, r=i, c=j: validate_entry(r, c))
            row.append(e)
        entries.append(row)

create_grid()

# ÜST PANEL (Zorluk ve Süre)
top_frame = tk.Frame(root)
top_frame.grid(row=11, column=0, columnspan=9, pady=5)

difficulty_var = tk.StringVar(value="Orta")
tk.Label(top_frame, text="Zorluk:").pack(side=tk.LEFT)
diff_menu = ttk.OptionMenu(top_frame, difficulty_var, "Orta", "Kolay", "Orta", "Zor")
diff_menu.pack(side=tk.LEFT, padx=5)

timer_label = tk.Label(top_frame, text="Süre: 00:00", font=('Arial', 12, 'bold'), fg="darkred")
timer_label.pack(side=tk.LEFT, padx=20)

# BUTONLAR
tk.Button(root, text="1. Fotoğraftan Tara", command=start_detect, bg="yellow", height=2).grid(row=9, column=0, columnspan=4, sticky="we")
tk.Button(root, text="2. Yapay Zeka Çözsün", command=animated_solve, bg="lightgreen", height=2).grid(row=9, column=4, columnspan=5, sticky="we")
tk.Button(root, text="Yeni Oyun Oluştur", command=start_new_game, bg="lightblue", height=2).grid(row=10, column=0, columnspan=4, sticky="we")
tk.Button(root, text="Temizle", command=lambda: [ (e.delete(0, tk.END), e.config(bg="#e0e0e0" if (r // 3 + c // 3) % 2 == 0 else "white")) for r, row in enumerate(entries) for c, e in enumerate(row)], bg="white", height=2).grid(row=10, column=4, columnspan=5, sticky="we")

root.mainloop()