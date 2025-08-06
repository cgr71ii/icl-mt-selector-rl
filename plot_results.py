
import sys

import matplotlib.pyplot as plt

output_path = sys.argv[1]
title = sys.argv[2] if len(sys.argv) > 2 else "Scores with Standard Deviation"

# Initialize data lists
x_vals = []
comet_vals, comet_stds = [], []
bleu_vals, bleu_stds = [], []
chrf2_vals, chrf2_stds = [], []

# Read from stdin
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parts = line.split('\t')
    if len(parts) != 4:
        raise ValueError(f"Invalid line: {line}")

    x = parts[0]

    def parse_value(value_str):
        try:
            value, std = value_str.split('+-')
            return float(value.strip()), float(std.strip())
        except:
            raise ValueError(f"Invalid value±stdev format: {value_str}")

    comet, comet_std = parse_value(parts[1])
    bleu, bleu_std = parse_value(parts[2])
    chrf2, chrf2_std = parse_value(parts[3])

    x_vals.append(x)
    comet_vals.append(comet)
    comet_stds.append(comet_std)
    bleu_vals.append(bleu)
    bleu_stds.append(bleu_std)
    chrf2_vals.append(chrf2)
    chrf2_stds.append(chrf2_std)

# Convert x values to something plottable if they are numeric
x_numeric = [float(x) for x in x_vals]

# Plot
plt.figure(figsize=(10, 6))

# COMET
plt.plot(x_numeric, comet_vals, label='COMET', color='blue')
plt.scatter(x_numeric, comet_vals, color='blue', s=30)
plt.fill_between(
    x_numeric,
    [v - s for v, s in zip(comet_vals, comet_stds)],
    [v + s for v, s in zip(comet_vals, comet_stds)],
    color='blue', alpha=0.2
)

# BLEU
plt.plot(x_numeric, bleu_vals, label='BLEU', color='green')
plt.scatter(x_numeric, bleu_vals, color='green', s=30)
plt.fill_between(
    x_numeric,
    [v - s for v, s in zip(bleu_vals, bleu_stds)],
    [v + s for v, s in zip(bleu_vals, bleu_stds)],
    color='green', alpha=0.2
)

# chrF2
plt.plot(x_numeric, chrf2_vals, label='chrF2', color='red')
plt.scatter(x_numeric, chrf2_vals, color='red', s=30)
plt.fill_between(
    x_numeric,
    [v - s for v, s in zip(chrf2_vals, chrf2_stds)],
    [v + s for v, s in zip(chrf2_vals, chrf2_stds)],
    color='red', alpha=0.2
)

plt.xlabel('ICL examples')
plt.ylabel('Score')
plt.title(title)
plt.legend()
plt.grid(True)

# Set xticks if categorical
if not all(isinstance(x, float) for x in x_numeric):
    plt.xticks(x_numeric, x_vals, rotation=45)

plt.tight_layout()

plt.savefig(output_path, dpi=300, bbox_inches='tight')
#print(f"Figure saved to {output_path}")
