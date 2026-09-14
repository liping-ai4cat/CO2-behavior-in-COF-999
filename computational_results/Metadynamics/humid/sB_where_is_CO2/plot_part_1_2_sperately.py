import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('species_analysis_sB.csv')

# Define the columns of interest and the renaming mapping
rename_map = {
    'count_NHCOOH': 'NHCOOH',
    'count_NHCOO-': 'NHCOO-',
    'count_NH2COO': 'NH2COO',
    'count_NCOO': 'NCOO',
    'count_NCOOH': 'NCOOH'
}

# Rename the columns
df = df.rename(columns=rename_map)

# List of the new column names to analyze
columns = ['NHCOOH', 'NHCOO-', 'NCOOH', 'NCOO']

# Group by partition and sum the columns
grouped_sums = df.groupby('partition')[columns].sum()

# --- MODIFIED SECTION START ---
# Calculate the GRAND TOTAL (sum of all species in Part 1 + Part 2)
grand_total = grouped_sums.sum().sum()

# Calculate the probability over the TOTAL number
# We divide every cell by the single grand_total value
probabilities = grouped_sums / grand_total
# --- MODIFIED SECTION END ---

# Print the results
print(f"Grand Total Count: {grand_total}")
print("Probabilities (Global):")
print(probabilities)

# Plot them out into a single grouped bar graph
# Note: The sum of ALL bars in the plot will now equal 1.0
ax = probabilities.T.plot(kind='bar', figsize=(6, 5), edgecolor='black', width=0.8)

# Add labels and title
plt.xlabel('Species')
plt.ylabel('Global Occurrence Probability')
plt.title('Species Probability (Normalized by Grand Total)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Partition')

# Add value labels on top of bars
for p in ax.patches:
    # Only label if height is greater than 0 to avoid clutter
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    fontsize=9, rotation=0, xytext=(0, 2),
                    textcoords='offset points')

plt.tight_layout()
plt.savefig('species_probability_global.png')
plt.show()
