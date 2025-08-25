"""
The APyT GUI forms module
=========================

This module provides a lightweight graphical user interface (GUI) for
interactive data entry forms using Tkinter. It is intended for situations where
command-line input is insufficient or where a simple, cross-platform dialog is
preferred.

Currently, the module focuses on login functionality, offering a secure and
user-friendly way to collect authentication credentials during runtime.


List of functions
-----------------

* :func:`login`
    Open a graphical login dialog to collect username and password.


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. codeauthor::    Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
#
#
__version__ = "0.1.0"
__all__ = ["login"]
#
#
# import modules
import logging
import tkinter as tk
#
#
# import individual functions
from os import getlogin
from tkinter import ttk
from ttkthemes import ThemedTk
#
#
#
#
# set up logger
logger = logging.getLogger(__name__)
#
#
#
#
################################################################################
#
# public functions
#
################################################################################
def login():
    """
    Open a graphical login dialog to collect username and password.

    A themed Tkinter window is created with input fields for the username and
    password, along with a "Login" button. The password field masks input
    characters. The dialog is centered on the screen and cannot be closed
    without entering credentials. Pressing the Enter key also submits the form.

    Returns
    -------
    tuple of (str, str)
        A 2-tuple containing the username and password entered by the user.
    """
    #
    #
    # create root window
    root = ThemedTk(theme = 'breeze')
    root.withdraw()
    #
    #
    # create sign-in frame
    pad = 10
    login = ttk.Frame(root)
    login.pack(padx = pad, pady = pad, fill = 'x', expand = True)
    user, password = tk.StringVar(login, getlogin()), tk.StringVar(login)
    #
    ttk.Label(login, text = "Please provide your login credentials."). \
        pack(anchor = 'w', pady = (0, pad))
    #
    ttk.Label(login, text = "User name:").pack(fill = 'x', expand = True)
    ttk.Entry(login, textvariable = user). \
        pack(fill = 'x', expand = True, pady = (0, pad))
    #
    ttk.Label(login, text = "Password:").pack(fill = 'x', expand = True)
    password_entry = ttk.Entry(login, textvariable = password, show = "*")
    password_entry.pack(fill = 'x', expand = True,  pady = (0, pad))
    password_entry.focus()
    #
    # login button
    ttk.Button(login, text = "Login", command = root.quit). \
        pack(fill = 'x', expand = True)
    #
    #
    # set root window properties
    root.resizable(False, False)
    root.title("Login")
    root.bind('<Return>', lambda x: root.quit())
    root.protocol('WM_DELETE_WINDOW', lambda: None)
    root.eval('tk::PlaceWindow . center')
    #
    #
    # get authorization credentials
    root.deiconify()
    root.mainloop()
    auth = (user.get(), password.get())
    root.destroy()
    #
    #
    # return tuple consisting of username/password
    return auth
