import tkinter as tk
from tkinter import ttk

def show_choice():
    print(f"Selected choice: {choice.get()}")

root = tk.Tk()
root.title("Tkinter Multiple Choice Menu")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Declare a Tkinter variable to store the selected choice
choice = tk.StringVar()

# Create radio buttons
ttk.Radiobutton(frame, text="Option 1", variable=choice, value="option1").grid(row=0, column=0, sticky=tk.W)
ttk.Radiobutton(frame, text="Option 2", variable=choice, value="option2").grid(row=1, column=0, sticky=tk.W)
ttk.Radiobutton(frame, text="Option 3", variable=choice, value="option3").grid(row=2, column=0, sticky=tk.W)

# Create a button to show the selected choice
ttk.Button(frame, text="Show Choice", command=show_choice).grid(row=3, column=0, pady=10)

root.mainloop()
