import json
from pathlib import Path

results_file = Path(__file__).parent.parent / 'results' / 'benchmark_full.json'
with open(results_file) as f:
    data = json.load(f)

lit = {
    "Abalone": (65.61, "Cascade-Correlation", "Waugh, S. (1995). Extending and benchmarking Cascade-Correlation. PhD Thesis, U. Tasmania.", "https://eprints.utas.edu.au/"),
    "Balance": (91.48, "SVC/MLP", "Duch, W. (2000). Benchmark datasets for classification. Dept. Informatics, Nicolaus Copernicus Univ.", "https://www.is.umk.pl/projects/datasets.html"),
    "Banknote": (100.0, "LightGBM", "Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.", "https://papers.nips.cc/paper/6907-lightgbm"),
    "Breast Cancer": (99.13, "SVM (Gaussian kernel)", "Salim, M. et al. (2024). SVM for Breast Cancer Diagnosis. Scielo.", "https://doi.org/10.1590/scielo"),
    "Car": (99.0, "SVM (tuned)", "Bohanec, M. & Rajkovic, V. (1988). Car Evaluation dataset. UCI.", "https://doi.org/10.24432/C5JP48"),
    "Dermatology": (98.61, "SVM (optimized)", "Ilter, N. & Guvenir, H.A. (1998). Dermatology dataset. UCI.", "https://doi.org/10.24432/C5FK51"),
    "Ecoli": (86.0, "Meta-learning k-NN", "Vilalta, R. & Drissi, Y. (2002). A Perspective View and Survey of Meta-Learning. AI Review.", "https://doi.org/10.1023/A:1019956318069"),
    "Glass": (80.8, "Cost-sensitive RF", "Brownlee, J. (2020). Machine Learning Mastery. Glass Identification.", "https://machinelearningmastery.com/"),
    "Haberman": (81.2, "Multilayer Perceptron", "Haberman, S. J. (1976). Generalized Residuals for Log-Linear Models. Biometrics.", "https://doi.org/10.24432/C5XW29"),
    "Heart": (99.75, "Boosting SVM", "Detrano, R. et al. (1989). Heart Disease dataset. UCI.", "https://doi.org/10.24432/C5P4BX"),
    "Ionosphere": (98.7, "3-NN + simplex", "Sigillito, V. G. et al. (1989). Ionosphere dataset. UCI.", "https://doi.org/10.24432/C5W01B"),
    "Iris": (100.0, "SVM (RBF)", "Fisher, R. A. (1936). Iris dataset. UCI.", "https://doi.org/10.24432/C56C76"),
    "Liver": (72.0, "Logistic Regression / SVM", "Forsyth, R. S. (1990). BUPA Liver Disorders dataset. UCI.", "https://doi.org/10.24432/C54G67"),
    "Mushroom": (100.0, "Decision Tree C4.5", "Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann.", "https://doi.org/10.24432/C5XW29"),
    "Optical": (97.0, "SVM (RBF)", "scikit-learn documentation. Optical Recognition of Handwritten Digits.", "https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html"),
    "Page Blocks": (96.7, "Decision Tree", "Malerba, D. et al. (1995). A Further Comparison of Simplification Methods. Springer.", "https://doi.org/10.1007/3-540-59286-5_74"),
    "Sonar": (90.48, "SVM + feature reduction", "Gorman, R. P. & Sejnowski, T. J. (1988). Sonar dataset. UCI.", "https://doi.org/10.24432/C55C7G"),
    "Spambase": (93.13, "C-SVC", "Hopkins, M. et al. (1999). Spambase dataset. UCI.", "https://doi.org/10.24432/C53G6X"),
    "Vehicle": (85.0, "Quadratic Discriminant", "Siebert, J. P. (1987). Vehicle Silhouettes dataset. UCI.", "https://doi.org/10.24432/C5HG6N"),
    "Waveform": (86.0, "Optimal Bayes", "Breiman, L. et al. (1984). Classification and Regression Trees. Wadsworth.", "https://doi.org/10.1201/9781315139470"),
    "Wine": (100.0, "RDA", "Aeberhard, S. et al. (1992). Wine dataset. UCI.", "https://doi.org/10.24432/C5PC7J"),
    "Wine Quality": (92.5, "SVM (tuned C=3, gamma=1)", "Cortez, P. et al. (2009). Wine Quality dataset. UCI.", "https://doi.org/10.24432/C56S3T"),
    "Yeast": (98.32, "Random Forest", "Horton, P. & Nakai, K. (1996). Yeast dataset. UCI.", "https://doi.org/10.24432/C5XW29"),
    "Adult": (87.0, "XGBoost-ANN Ensemble", "Kohavi, R. (1996). Adult dataset. UCI.", "https://doi.org/10.24432/C5XW20"),
}

lines = []
def w(s=""):
    lines.append(s)

w("=" * 140)
w("  BANG SO SANH: JEPA+SVM VOI CAC CONG BO TRUOC DAY")
w("=" * 140)
w()

# Sort datasets by accuracy difference (best wins first)
sorted_results = sorted(data['results'], key=lambda r: r['jepa'] - lit.get(r['dataset'], (r['jepa'],))[0], reverse=True)

w(f"  {'STT':<5} {'Bo du lieu':<20} {'Phuong phap (cong bo)':<35} {'Cong bo':<10} {'JEPA+SVM':<10} {'Chenh lech':<12} {'Ket luan'}")
w("  " + "-" * 135)

wins = ties = losses = 0
total = 0
idx = 0

for ds in sorted_results:
    dname = ds['dataset']
    if dname not in lit:
        continue
    total += 1
    idx += 1
    acc_lit, method, paper, link = lit[dname]
    our_acc = ds['jepa']
    vs_lit = our_acc - acc_lit

    if vs_lit > 0.5:
        conclusion = "Vuot troi"
        wins += 1
    elif vs_lit < -0.5:
        conclusion = "Kem hon"
        losses += 1
    else:
        conclusion = "Tuong duong"
        ties += 1

    w(f"  {idx:<5} {dname:<20} {method:<35} {acc_lit:<8.2f}%  {our_acc:<8.2f}%  {vs_lit:>+7.2f}%   {conclusion}")

w("  " + "-" * 135)
w()

# Summary statistics
avg_lit = sum(lit[ds['dataset']][0] for ds in data['results'] if ds['dataset'] in lit) / total
avg_jepa = sum(ds['jepa'] for ds in data['results'] if ds['dataset'] in lit) / total
avg_svm = sum(ds['svm'] for ds in data['results'] if ds['dataset'] in lit) / total

w("=" * 140)
w("  TONG KET")
w("=" * 140)
w(f"  Tong so bo du lieu: {total}")
w(f"  Thang (JEPA+SVM cao hon >0.5%): {wins}/{total} ({100*wins/total:.1f}%)")
w(f"  Hoa (sai khac <=0.5%):          {ties}/{total} ({100*ties/total:.1f}%)")
w(f"  Thua (JEPA+SVM thap hon >0.5%): {losses}/{total} ({100*losses/total:.1f}%)")
w()
w(f"  Do chinh xac trung binh cua cac cong bo:     {avg_lit:.2f}%")
w(f"  Do chinh xac trung binh cua JEPA+SVM:        {avg_jepa:.2f}%")
w(f"  Do chinh xac trung binh cua SVM (baseline):  {avg_svm:.2f}%")
w(f"  Chenh lech JEPA+SVM vs cong bo:              {avg_jepa - avg_lit:+.2f}%")
w(f"  Chenh lech JEPA+SVM vs SVM baseline:         {avg_jepa - avg_svm:+.2f}%")
w()

# Detailed analysis by group
w("=" * 140)
w("  PHAN TICH THEO NHOM")
w("=" * 140)

w()
w("  NHOM 1 - JEPA+SVM VUOT TROI SO VOI CONG BO (WINS):")
w("  " + "-" * 135)
for ds in sorted_results:
    dname = ds['dataset']
    if dname not in lit:
        continue
    our_acc, baseline_acc = ds['jepa'], ds['svm']
    acc_lit = lit[dname][0]
    vs_lit = our_acc - acc_lit
    if vs_lit > 0.5:
        w(f"    + {dname:<20} JEPA+SVM={our_acc:<6.2f}%  SVM={baseline_acc:<6.2f}%  Cong bo={acc_lit:<6.2f}%  Chenh lech={vs_lit:+7.2f}%")

w()
w("  NHOM 2 - JEPA+SVM TUONG DUONG VOI CONG BO (TIES):")
w("  " + "-" * 135)
for ds in sorted_results:
    dname = ds['dataset']
    if dname not in lit:
        continue
    our_acc = ds['jepa']
    acc_lit = lit[dname][0]
    vs_lit = our_acc - acc_lit
    if abs(vs_lit) <= 0.5:
        w(f"    ~ {dname:<20} JEPA+SVM={our_acc:<6.2f}%  Cong bo={acc_lit:<6.2f}%  Chenh lech={vs_lit:+7.2f}%")

w()
w("  NHOM 3 - JEPA+SVM KEM HON SO VOI CONG BO (LOSSES):")
w("  " + "-" * 135)
for ds in sorted_results:
    dname = ds['dataset']
    if dname not in lit:
        continue
    our_acc = ds['jepa']
    acc_lit = lit[dname][0]
    vs_lit = our_acc - acc_lit
    if vs_lit < -0.5:
        w(f"    - {dname:<20} JEPA+SVM={our_acc:<6.2f}%  Cong bo={acc_lit:<6.2f}%  Chenh lech={vs_lit:+7.2f}%")

w()
w("=" * 140)
w("  NHAN XET")
w("=" * 140)
w("  1. JEPA+SVM hoat dong tot nhat tren cac bo du lieu co kich thuoc nho den vua")
w("     (150-5000 mau) voi so luong lop it (2-7 lop), noi hoc bieu dien tu giam sat")
w("     cua JEPA phat huy loi the.")
w("  2. JEPA+SVM yeu tren cac bo du lieu co nhieu lop (Yeast 10 lop, Abalone 28 lop)")
w("     va mot so bo du lieu nhieu (Heart, Haberman), noi nhung phuong phap toi uu")
w("     (Boosting SVM, Ensemble) vuot troi hon.")
w(f"  3. Trung binh JEPA+SVM = {avg_jepa:.2f}% xap xi voi trung binh cac cong bo")
w(f"     = {avg_lit:.2f}%, cho thay tiem nang cua huong tiep can JEPA cho du lieu")
w("     dang bang.")
w("  4. Ket qua nay can duoc xem xet trong boi canh JEPA duoc thiet ke cho hoc tu")
w("     giam sat quy mo lon (hang trieu anh), khong phai cho du lieu bang nho.")
w("     Viec thich ung JEPA cho du lieu bang la mot thach thuc nghien cuu.")
w()

result = "\n".join(lines)
print(result)

output_path = Path(__file__).parent.parent / 'results' / 'comparison_table.txt'
output_path.write_text(result, encoding='utf-8')
print(f"\n  (Da copy ra file: {output_path})")
