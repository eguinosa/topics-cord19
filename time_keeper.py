# Gelin Eguinosa Rosique
# 2022

import time


class TimeKeeper:
    """
    Class to keep track of the time the programs expends processing the
    information. It doesn't count the time the program expends waiting for input
    from the user.
    """

    def __init__(self):
        """
        Create the basic variables to keep track of the running time of the
        program.
        """
        # Initialize variables
        self.start_time = time.time()
        self.runtime = 0
        # Set the state of the recording
        self.current_state = 'running'

    def pause(self):
        """
        Stop recording the time, until the user commands it to start recording
        again.
        """
        # Check if the TimeKeeper is currently running, otherwise we don't need
        # to do anything.
        if self.current_state == 'running':
            # Update the total runtime of the program.
            self.runtime += time.time() - self.start_time
            # Update the state of the recording
            self.current_state = 'paused'

    def restart(self):
        """
        Resets the start time of the TimeKeeper, either if it is currently on
        pause or running. As a result, it continues recording if it was on
        pause, or resets the time if it was currently running.
        """
        # Reset the value of the start time
        self.start_time = time.time()
        # Reset the runtime value
        self.runtime = 0
        # Update the state of the timer
        self.current_state = 'running'

    def total_runtime(self):
        """
        Get the total runtime of the timer at the current moment.
        """
        # Update the runtime value only is the TimeKeeper was running, otherwise
        # it's not necessary.
        if self.current_state == 'running':
            self.runtime += time.time() - self.start_time
            # Reset the start time, to avoid adding the same segment again
            self.start_time = time.time()
        
        # Return the updated runtime
        return self.runtime

    def formatted_runtime(self):
        """
        Transforms the elapsed time from the start of the program to a new format
        in hours, minutes and seconds.

        Returns: A string containing the elapsed time in <hours:minutes:seconds>
        """
        # Get the runtime at the current moment:
        current_runtime = self.total_runtime()

        # Calculating the hours, minutes, seconds and milliseconds
        hours = int(current_runtime / 3600)
        minutes = int((current_runtime - hours * 3600) / 60)
        seconds = int(current_runtime - hours * 3600 - minutes * 60)
        milliseconds = int((current_runtime - int(current_runtime)) * 1000)

        return f'{hours} h : {minutes} min : {seconds} sec : {milliseconds} mill'


# Testing the TimeKeeper
if __name__ == '__main__':
    stopwatch = TimeKeeper()
    time.sleep(3)
    print(stopwatch.formatted_runtime())
    time.sleep(4.5)
    print(stopwatch.formatted_runtime())
    time.sleep(62.5)
    print(stopwatch.formatted_runtime())
