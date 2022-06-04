# Gelin Eguinosa Rosique

from sys import stdout


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

