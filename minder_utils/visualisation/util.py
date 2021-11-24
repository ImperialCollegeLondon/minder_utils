def time_to_seconds(timestr):
    ftr = [3600, 60, 1]
    return sum([a * b for a, b in zip(ftr, map(int, timestr.split(':')))])
