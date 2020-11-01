import tkinter as tk
import tkinter.ttk as ttk
from typing import Callable, Literal, Optional, Tuple


# parent
def create_parent(title: str = None, *, resizable: Optional[Tuple[bool, bool]] = (True, True)):
    parent = tk.Tk()

    if title:
        parent.title(title)

    parent.resizable(resizable[0], resizable[1])

    return parent


# variables
def str_var(parent: tk.Misc, initial_value: str = None):
    return tk.StringVar(parent, initial_value)


def int_var(parent: tk.Misc, initial_value: int = None):
    return tk.IntVar(parent, initial_value)


def float_var(parent: tk.Misc, initial_value: float = None):
    return tk.DoubleVar(parent, initial_value)


def bool_var(parent: tk.Misc, initial_value: bool = None):
    return tk.BooleanVar(parent, initial_value)


# widgets
def frame(parent: tk.Misc, *, col: int = None, row: int = None, col_span: int = None,
          row_span: int = None, padx: int = 0, pady: int = 0, sticky: str = ''):
    frame_widget = ttk.Frame(parent)
    frame_widget.grid(column=col, row=row, columnspan=col_span, rowspan=row_span, padx=padx,
                      pady=pady, sticky=sticky)
    return frame_widget


def responsive_frame(parent: tk.Misc, *, orient: Literal["x", "y", "both"] = 'x', padx: int = 0,
                     pady: int = 0):
    frame_widget = ttk.Frame(parent)
    frame_widget.pack(fill=orient, padx=padx, pady=pady)
    return frame_widget


def responsive_named_frame(parent: tk.Misc, name: str, *, orient: Literal["x", "y", "both"] = 'x',
                           padx: int = 0, pady: int = 0):
    frame_widget = ttk.LabelFrame(parent, text=name)
    frame_widget.pack(fill=orient, padx=padx, pady=pady)
    return frame_widget


def responsive_text_label(parent: tk.Misc, text: str, *, padx: int = 0, pady: int = 0):
    label_widget = ttk.Label(parent, text=text)
    label_widget.pack(padx=padx, pady=pady)


def responsive_variable_label(parent: tk.Misc, var: tk.Variable, *, padx: int = 0, pady: int = 0):
    label_widget = ttk.Label(parent, textvariable=var)
    label_widget.pack(padx=padx, pady=pady)


def responsive_item_label(parent: tk.Misc, text: str, var: tk.Variable, *, padx: int = 0,
                          pady: int = 0):
    child_frame = ttk.Frame(parent)
    child_frame.pack(fill='x', padx=padx, pady=pady)
    label_widget = ttk.Label(child_frame, text=text)
    label_widget.pack(side='left')
    value_widget = ttk.Label(child_frame, textvariable=var)
    value_widget.pack(side='right')


def responsive_button(parent: tk.Misc, text: str, action: Callable, *, padx: int = 0,
                      pady: int = 0):
    button_widget = ttk.Button(parent, text=text, command=action)
    button_widget.pack(fill='x', padx=padx, pady=pady)
