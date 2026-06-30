"""
Save benchmark results to JSON and display summary.
"""
import json, time
from pathlib import Path

RESULT_FILE = Path(__file__).parent.parent / 'results' / 'benchmark_full.json'
RESULT_FILE.parent.mkdir(exist_ok=True)

results = [
    {"dataset":"Abalone","n":4177,"f":8,"c":28,"svm":30.38,"jepa":30.38,"impr":0.00,"best":"Tie","time":158},
    {"dataset":"Balance","n":625,"f":4,"c":3,"svm":91.20,"jepa":98.40,"impr":7.89,"best":"JEPA","time":58},
    {"dataset":"Banknote","n":1372,"f":4,"c":2,"svm":100.00,"jepa":100.00,"impr":0.00,"best":"Tie","time":74},
    {"dataset":"Breast Cancer","n":569,"f":30,"c":2,"svm":98.25,"jepa":98.25,"impr":0.00,"best":"Tie","time":56},
    {"dataset":"Car","n":1728,"f":6,"c":4,"svm":96.24,"jepa":97.40,"impr":1.20,"best":"JEPA","time":99},
    {"dataset":"Dermatology","n":366,"f":34,"c":6,"svm":95.95,"jepa":95.95,"impr":0.00,"best":"Tie","time":46},
    {"dataset":"Ecoli","n":336,"f":8,"c":8,"svm":98.53,"jepa":100.00,"impr":1.49,"best":"JEPA","time":40},
    {"dataset":"Glass","n":214,"f":10,"c":6,"svm":90.70,"jepa":95.35,"impr":5.13,"best":"JEPA","time":27},
    {"dataset":"Haberman","n":306,"f":3,"c":2,"svm":67.74,"jepa":67.74,"impr":0.00,"best":"Tie","time":36},
    {"dataset":"Heart","n":303,"f":13,"c":5,"svm":57.38,"jepa":57.38,"impr":0.00,"best":"Tie","time":36},
    {"dataset":"Ionosphere","n":351,"f":34,"c":2,"svm":95.77,"jepa":95.77,"impr":0.00,"best":"Tie","time":41},
    {"dataset":"Iris","n":150,"f":4,"c":3,"svm":96.67,"jepa":100.00,"impr":3.45,"best":"JEPA","time":18},
    {"dataset":"Liver","n":345,"f":6,"c":2,"svm":69.57,"jepa":75.36,"impr":8.33,"best":"JEPA","time":40},
    {"dataset":"Mushroom","n":8124,"f":22,"c":2,"svm":100.00,"jepa":100.00,"impr":0.00,"best":"Tie","time":300},
    {"dataset":"Optical","n":5620,"f":64,"c":10,"svm":99.20,"jepa":99.20,"impr":0.00,"best":"Tie","time":216},
    {"dataset":"Page Blocks","n":5473,"f":10,"c":5,"svm":94.34,"jepa":96.44,"impr":2.23,"best":"JEPA","time":202},
    {"dataset":"Sonar","n":208,"f":60,"c":2,"svm":83.33,"jepa":92.86,"impr":11.43,"best":"JEPA","time":27},
    {"dataset":"Spambase","n":4601,"f":57,"c":2,"svm":92.07,"jepa":92.83,"impr":0.83,"best":"JEPA","time":172},
    {"dataset":"Vehicle","n":846,"f":18,"c":4,"svm":71.76,"jepa":82.94,"impr":15.57,"best":"JEPA","time":82},
    {"dataset":"Waveform","n":5000,"f":21,"c":3,"svm":85.60,"jepa":85.90,"impr":0.35,"best":"JEPA","time":186},
    {"dataset":"Wine","n":178,"f":13,"c":3,"svm":100.00,"jepa":100.00,"impr":0.00,"best":"Tie","time":22},
    {"dataset":"Wine Quality","n":1599,"f":11,"c":2,"svm":74.38,"jepa":77.19,"impr":3.78,"best":"JEPA","time":88},
    {"dataset":"Yeast","n":1484,"f":9,"c":10,"svm":56.57,"jepa":59.60,"impr":5.36,"best":"JEPA","time":87},
]

meta = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "seed": 42,
    "protocol": "80/20 stratified holdout, MinMaxScaler(-1,1)",
    "svm_config": "SVC(kernel=rbf, C=1.0, gamma=scale)",
    "jepa_config": "embedding_dim=max(8, min(features*2, 32)), supervised fine-tune",
    "total_datasets": len(results),
    "wins": sum(1 for r in results if r['best'] == 'JEPA'),
    "ties": sum(1 for r in results if r['best'] == 'Tie'),
    "losses": sum(1 for r in results if r['best'] == 'SVM'),
    "avg_svm": sum(r['svm'] for r in results) / len(results),
    "avg_jepa": sum(r['jepa'] for r in results) / len(results),
    "avg_impr": (sum(r['jepa'] for r in results) - sum(r['svm'] for r in results)) / sum(r['svm'] for r in results) * 100,
}

full = {"meta": meta, "results": results}
RESULT_FILE.write_text(json.dumps(full, indent=2))
print(f"Saved to {RESULT_FILE}")
print(f"Wins={meta['wins']}/{meta['total_datasets']} Ties={meta['ties']}/{meta['total_datasets']} Losses={meta['losses']}/{meta['total_datasets']}")
print(f"Avg SVM={meta['avg_svm']:.2f}% Avg JEPA={meta['avg_jepa']:.2f}% Avg Impr={meta['avg_impr']:.2f}%")
