import tkinter as tk
import ttkbootstrap as ttk

context_dict = dict()

# Function to handle user input and generate response
def on_submit():
    user_input = chat_entry.get()
    if user_input:
        chat_display.config(state=tk.NORMAL)
        chat_display.insert(tk.END, f"You: {user_input}\n")
        chat_display.config(state=tk.DISABLED)

        # Generate response from Ollama
        # try:
        #     response_text, metadata = execute_ollama_rag(user_input)
        #     create_context_dict(metadata)
        #     # Insert Keys to middle frame
        #     for id in context_dict.keys():
        #         chunk_listbox.insert(tk.END, id)            
        # except TypeError:
        #     response_text = "No relevant information found."
            
        # # Display user input and response in the chat window
        # chat_display.config(state=tk.NORMAL)
        # chat_display.insert(tk.END, f"AI: {response_text}\n\n")
        # chat_display.config(state=tk.DISABLED)
        # chat_entry.delete(0, tk.END)

# Dummy function to simulate chunk filenames and chunk text
def create_context_dict(metadata):
    for id, doc in zip(metadata['ids'], metadata['documents']):
        # Slicing the filename and chunk #
        f_start_pos = id.rfind("\\")
        file_info = f'{id[f_start_pos + 1:]}'
        f_end_pos = file_info.find(":")
        filename = f'{file_info[-1]}:{file_info[:f_end_pos]}'
        context_dict[filename] = doc


def display_selected_chunk(event):
    try:
        selected_filename = chunk_listbox.get(chunk_listbox.curselection())
        chunk_text = context_dict[selected_filename]
        chunk_text_display.config(state=tk.NORMAL)
        chunk_text_display.delete(1.0, tk.END)
        chunk_text_display.insert(tk.END, chunk_text)
        chunk_text_display.config(state=tk.DISABLED)
    except tk.TclError:
        pass


# Set up the main application window
root = root = ttk.Window(themename="journal")  # You can change the theme here
root.title("Ollama Code Generator")

# Left column (Chat with AI)
chat_frame = ttk.Frame(root, padding=(10 ,10, 10, 10))
chat_frame.grid(row=0, column=0, sticky="nsew")

chat_label = ttk.Label(chat_frame, text="Chat with AI", background="#eb6864",foreground='white',anchor='center')
chat_label.pack(expand=True, fill='x',anchor='n')

chat_display = tk.Text(chat_frame, height=20, width=40, state=tk.DISABLED)
chat_display.pack(expand=True, fill='both')

chat_entry = tk.Entry(chat_frame, width=40)
chat_entry.pack(expand=True, fill='x', pady=(5,0))

submit_button = tk.Button(chat_frame, text="Submit", command=on_submit)
submit_button.pack(expand=True, fill='x', pady=(5,0))

# Middle column (Chunk filenames)
chunk_frame = ttk.Frame(root, padding=(10 ,10, 10, 10))
chunk_frame.grid(row=0, column=1, sticky="nsew")

chunk_label = ttk.Label(chunk_frame, text="Chunk Filenames", background="#eb6864",foreground='white',anchor='center')
# chunk_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
chunk_label.pack(expand=True, fill='x',anchor='n')

chunk_listbox = tk.Listbox(chunk_frame, height=20, width=20)
# chunk_listbox.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
chunk_listbox.pack(expand=True, fill='both',anchor='n')

chunk_listbox.bind("<<ListboxSelect>>", display_selected_chunk)

# Right column (Chunk text)
chunk_text_frame = ttk.Frame(root, padding=(10 ,10, 10, 10))
chunk_text_frame.grid(row=0, column=2, sticky="nsew")

chunk_text_label = ttk.Label(chunk_text_frame, text="Chunk Text", background="#eb6864",foreground='white',anchor='center')
chunk_text_label.pack(expand=True, fill='x',anchor='n')

chunk_text_display = tk.Text(chunk_text_frame, height=20, width=40, state=tk.DISABLED)
chunk_text_display.pack(expand=True, fill='both')

# Configure resizing behavior
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_rowconfigure(0, weight=1)

# Run the application
root.mainloop()
