import csv


def export_csv(path, times, counts, currents):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "particle_count", "current_a"])
        for time_s, count, current in zip(times, counts, currents):
            writer.writerow([time_s, count, current])
