import os
import pandas as pd
import matplotlib.pyplot as plt

ref_dir = "./ref/"

# for each file in ref_dir
for filename in os.listdir(ref_dir):
    if not filename.endswith('.csv'):
        continue

    ref_name = os.path.join(ref_dir, filename)
    ref = pd.read_csv(ref_name)
    out = pd.read_csv(os.path.join("./out/", filename))

    # extract all columns except the first and plot each one in a separate subplot of size (2,4)
    # ignore 1 header line
    plt.figure(figsize=(15, 8))

    cols = ref.columns[1:]
    m = max(max(ref[cols].max()), max(out[cols].max()))
    for i, col in enumerate(ref.columns[1:]):
        plt.subplot(2, 4, i + 1)
        plt.plot(ref[col], label='Reference', color='blue')
        plt.plot(out[col], label='Test', color='orange')
        plt.title(col)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.ylim([0,m])
    plt.show()