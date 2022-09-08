# Gelin Eguinosa Rosique
# 2022

from sys import stdout
from time import time, ctime


def progress_bar(progress, total):
    """
    Print on the console a bar representing the progress of the function that
    called this method.

    Args:
        progress: How many of the planned actions the program has done.
        total: The total amount of actions the program needs to do.
    """
    # Values to print.
    steps = progress * 40 // total
    percentage = progress * 100 // total

    # Print progress bar.
    stdout.write('\r')
    stdout.write("[%-40s] %03s%%" % ('=' * steps, percentage))
    stdout.flush()

    # Add Break at the end of the progress bar, if we are done.
    if progress == total:
        print()


def progress_msg(message):
    """
    Display the provided 'message' to the user about the progress of the
    program.

    Args:
        message: String with the message we have to display.
    """
    print(f"[{time_info()}] {message}")


def time_info():
    """
    Create a string with the time data in the format hh:mm:ss.mss.

    Returns: String with the time data.
    """
    current_time = time()
    millisecond = int((current_time - int(current_time)) * 1000)
    # Format HH:MM:SS.MSS
    time_data = ctime(current_time)[11:19] + '.' + number_to_3digits(millisecond)
    return time_data


def big_number(number):
    """
    Add commas to number with more than 3 digits, so they are more easily read.

    Args:
        number: The number we want to transform to string

    Returns:
        The string of the number with the format: dd,ddd,ddd,ddd
    """
    # Get the string of the number.
    number_string = str(number)

    # Return its string if it's not big enough.
    if len(number_string) <= 3:
        return number_string

    # Add the commas.
    new_string = number_string[-3:]
    number_string = number_string[:-3]
    while len(number_string) > 0:
        new_string = number_string[-3:] + ',' + new_string
        number_string = number_string[:-3]

    # Return the reformatted string of the number.
    return new_string


def number_to_3digits(number):
    """
    Transform a number smaller than 1000 (0-999) to a string representation with
    three characters (000, 001, ..., 021, ..., 089, ..., 123, ..., 999).
    """
    # Make sure the value we transform is under 1000 and is positive.
    mod_number = number % 1000
    
    if mod_number < 10:
        return "00" + str(mod_number)
    elif mod_number < 100:
        return "0" + str(mod_number)
    else:
        return str(mod_number)
