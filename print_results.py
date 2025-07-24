
import sys
import statistics

results = []

for f in sys.argv[1:]:
    with open(f, 'r') as fd:
        r1 = []

        for l in fd:
            l = l.strip()

            r1.append(l)

        assert len(r1) > 3

        r1 = r1[-3:]

        assert r1[0].startswith("System score: ")

        r1[0] = r1[0].replace("System score: ", "").strip()
        r1[0] = round(float(r1[0]) * 100, 2)
        r1[1] = float(r1[1].strip())
        r1[2] = float(r1[2].strip())

        results.append(tuple(r1))

# statistics across columns
avg = [round(statistics.mean(x), 2) for x in zip(*results)]
stdev = [round(statistics.stdev(x), 2) for x in zip(*results)]
results_format = [f"{x} +- {y}" for x, y in zip(avg, stdev)]
all_results_format = " | ".join(results_format)

print(all_results_format)
