# Gelin Eguinosa Rosique

from sys import stdout


def progress_bar(progress, total):
    """
    Print on the console a bar representing the progress of the function that
    called this method.
    :progress: How many actions the program has already covered.
    :total: The total amount of actions the program need to do.
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
