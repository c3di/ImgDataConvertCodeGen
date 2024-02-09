def get_execution_time(begin_time, end_time):
    total_seconds = (end_time - begin_time).total_seconds()

    # Convert the execution time to days, hours, minutes, and seconds
    days = total_seconds // (24 * 3600)
    total_seconds = total_seconds % (24 * 3600)
    hours = total_seconds // 3600
    total_seconds %= 3600
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    return f"The execution time is {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, and {seconds} seconds."
