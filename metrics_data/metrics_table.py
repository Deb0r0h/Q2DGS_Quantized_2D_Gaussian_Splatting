import os
import pandas as pd
import yaml

file_path = "../output/date"

data = []

for scan_folder in os.listdir(file_path):
    metrics_path = os.path.join(file_path, scan_folder,"train","ours_30000","metrics.yml")
    if os.path.exists(metrics_path):
        with open(metrics_path,"r") as file:
            metrics = yaml.safe_load(file)
        data.append({"scan":scan_folder,
                     "f_score":metrics.get("f_score",None),
                     "Chamfer":metrics.get("chamfer_dist",None),
                     "MSE": metrics.get("MSE", None)
                     })
    else:
        print("No file metrics.yml in {}".format(scan_folder))

df = pd.DataFrame(data)
#print(df.to_string(index=False))

# to export the latex table
latex_table = df.to_latex(index=False,header=True,caption="Metrics")
with open("metrics_table.tex", "w") as file:
    file.write(latex_table)

print("Latex table saved to metrics_table.tex")