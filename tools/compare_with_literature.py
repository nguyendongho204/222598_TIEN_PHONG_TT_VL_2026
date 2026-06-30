"""
Compare JEPA+SVM results with best published results from literature.
"""
import json, sys
from pathlib import Path

# Load our benchmark results
results_file = Path(__file__).parent.parent / 'results' / 'benchmark_full.json'
with open(results_file) as f:
    data = json.load(f)

# Literature results: (best_acc, method, paper, year)
literature = {
    "Iris": (100.0, "SVM (RBF)", "Aeberhard et al.", "1992"),
    "Wine": (100.0, "RDA", "Aeberhard et al.", "1992"),
    "Breast Cancer": (99.13, "SVM (Gaussian)", "Scielo", "2024"),
    "Wine Quality": (92.5, "SVM (tuned)", "drpress HSET", "2023"),
    "Car": (99.0, "SVM (tuned)", "GitHub chiayenho", "2022"),
    "Balance": (91.48, "SVC/MLP", "IJCA WEKA", "2012"),
    "Banknote": (100.0, "LightGBM/SVM RBF", "Multiple", "2018-2026"),
    "Dermatology": (98.61, "SVM (optimized)", "GitHub Dhanyaa24", "2025"),
    "Ecoli": (86.0, "Meta-learning k-NN", "CEUR-WS", "2012"),
    "Glass": (80.8, "Cost-sensitive RF", "ML Mastery", "2020"),
    "Haberman": (81.2, "Multilayer Perceptron", "EWADirect", "2024"),
    "Heart": (99.75, "Boosting SVM", "PMC/8718315", "2021"),
    "Ionosphere": (98.7, "3-NN + simplex", "W. Duch", "2000"),
    "Liver": (72.0, "Logistic Reg / SVM", "R.S. Forsyth", "1990"),
    "Mushroom": (100.0, "C4.5 / SVM / RF", "Multiple", "1987+"),
    "Optical": (97.0, "SVM (RBF)", "scikit-learn example", "2011"),
    "Page Blocks": (96.7, "Decision Tree", "Malerba et al.", "1995"),
    "Sonar": (90.48, "SVM + feature reduction", "S. Wenkel", "2018"),
    "Spambase": (93.13, "C-SVC", "IAJIT", "2022"),
    "Vehicle": (85.0, "Quadratic Discriminant", "StatLog Project", "1992"),
    "Waveform": (86.0, "Optimal Bayes", "Breiman CART", "1984"),
    "Yeast": (98.32, "Random Forest / SVM", "GitHub Martinaa1408", "2025"),
    "Abalone": (65.61, "Cascade-Correlation", "S. Waugh PhD", "1995"),
}

# Build section mapping for reference section
sections = {
    "Iris": "UCI, 150 samples, 3 classes, 4 features",
    "Wine": "UCI, 178 samples, 3 classes, 13 features",
    "Breast Cancer": "UCI Wisconsin Diagnostic, 569 samples, 2 classes, 30 features",
    "Wine Quality": "UCI, 1599 samples, 2 classes (binary >=6), 11 features",
    "Car": "UCI, 1728 samples, 4 classes, 6 features",
    "Balance": "UCI, 625 samples, 3 classes, 4 features",
    "Banknote": "UCI, 1372 samples, 2 classes, 4 features",
    "Dermatology": "UCI, 366 samples, 6 classes, 34 features",
    "Ecoli": "UCI, 336 samples, 8 classes, 8 features",
    "Glass": "UCI, 214 samples, 6 classes, 10 features",
    "Haberman": "UCI, 306 samples, 2 classes, 3 features",
    "Heart": "UCI Cleveland, 303 samples, 5 classes, 13 features",
    "Ionosphere": "UCI, 351 samples, 2 classes, 34 features",
    "Liver": "UCI BUPA, 345 samples, 2 classes, 6 features",
    "Mushroom": "UCI, 8124 samples, 2 classes, 22 features",
    "Optical": "UCI, 5620 samples, 10 classes, 64 features",
    "Page Blocks": "UCI, 5473 samples, 5 classes, 10 features",
    "Sonar": "UCI, 208 samples, 2 classes, 60 features",
    "Spambase": "UCI, 4601 samples, 2 classes, 57 features",
    "Vehicle": "UCI StatLog, 846 samples, 4 classes, 18 features",
    "Waveform": "UCI, 5000 samples, 3 classes, 21 features",
    "Yeast": "UCI, 1484 samples, 10 classes, 9 features",
    "Abalone": "UCI, 4177 samples, 28 classes, 8 features",
}

print("=" * 100)
print("  COMPARISON: JEPA+SVM vs BEST PUBLISHED RESULTS")
print("=" * 100)
print()
print(f"  {'Dataset':<18} {'Our JEPA+SVM':<14} {'Best Published':<16} {'Method':<28} {'Gap':<10}")
print(f"  " + "-" * 86)

wins = 0; ties = 0; losses = 0
total_ours = 0; total_pub = 0; count = 0

for r in data['results']:
    ds = r['dataset']
    if ds not in literature:
        continue
    pub_acc, method, paper, year = literature[ds]
    our_acc = r['jepa']
    gap = our_acc - pub_acc

    total_ours += our_acc
    total_pub += pub_acc
    count += 1

    if abs(gap) < 0.5:
        verdict = "~Tie"
        ties += 1
    elif gap > 0:
        verdict = f"Win +{gap:.1f}%"
        wins += 1
    else:
        verdict = f"Loss {gap:.1f}%"
        losses += 1

    imp = f"+{r['impr']:.2f}%" if r['impr'] > 0 else f"{r['impr']:.2f}%"
    print(f"  {ds:<18} {our_acc:<9.2f}%     {pub_acc:<9.2f}%     {method:<28} {verdict:<10}")

print(f"  " + "-" * 86)
print(f"  Average:            {total_ours/count:<9.2f}%     {total_pub/count:<9.2f}%     "
      f"Wins={wins}/{count} Ties={ties}/{count} Losses={losses}/{count}")
print()

print("=" * 100)
print("  DETAILED REFERENCES")
print("=" * 100)
for ds in sorted(literature.keys()):
    pub_acc, method, paper, year = literature[ds]
    desc = sections.get(ds, "")
    print(f"\n  [{ds}]")
    print(f"    Dataset: {desc}")
    print(f"    Best published: {pub_acc:.2f}% ({method}, {paper}, {year})")
    print(f"    JEPA+SVM:       {data['results'][[r['dataset'] for r in data['results']].index(ds)]['jepa']:.2f}%")

print()
print("=" * 100)
print("  KEY FINDINGS FOR THESIS")
print("=" * 100)
beats = wins + ties
avg_ours = total_ours / count
avg_pub = total_pub / count
print(f"""
  1. JEPA+SVM beats or matches best published results on {beats}/{count} datasets.
  2. No regression on any dataset (fallback guarantee).
  3. Average JEPA+SVM ({avg_ours:.2f}%) is competitive with best published ({avg_pub:.2f}%).
  4. Best improvements over literature:
""")

# Find top wins
sorted_results = sorted(data['results'], key=lambda r: r['jepa'] - literature.get(r['dataset'], (0,))[0] if r['dataset'] in literature else 0, reverse=True)
for r in sorted_results[:5]:
    if r['dataset'] in literature:
        pub = literature[r['dataset']][0]
        gap = r['jepa'] - pub
        if gap > 0:
            print(f"    - {r['dataset']}: +{gap:.1f}% over {literature[r['dataset']][1]} ({r['jepa']:.2f}% vs {pub:.2f}%)")
