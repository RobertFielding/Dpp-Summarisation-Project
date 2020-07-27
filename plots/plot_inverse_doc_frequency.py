import matplotlib
import matplotlib.pyplot as plt
import json

inv_freq_file = "train_na=5_inv_doc_freq.json"

with open(inv_freq_file) as f:
    inv_freq_dict = json.load(f)

fig, axes = plt.subplots(1, 2, fgsize=(20, 10))

inv_freq_list = sorted(list(inv_freq_dict.items()), key=lambda x: x[1])

ax = axes[0]
x, y = zip(*inv_freq_list[:10])
x, y = list(x), list(y)
x = ["{}\n{}".format(word, rank) for rank, word in enumerate(x, 1)]
x.append("...")
y.append(0)
ax.bar(x, y)
ax.set_ylabel("Inverse Document Frequency")
ax.set_title("10 Most Frequent Words")

ax = axes[1]
n=20_000
x, y = zip(*inv_freq_list[n:n+10])
x, y = list(x), list(y)
x = ["{}\n{:,}".format(word, rank) for rank, word in enumerate(x, n+1)]
x = ["..."] + x + ["...."]
y = [0] + y + [0]
ax.bar(x, y)
ax.set_ylabel("Inverse Document Frequency")
matplotlib.pyplot.sca(ax)
ax.set_title(f"Words ranking between {n+1:,} and {n+10:,}")
plt.xticks(rotation=45)

fig.suptitle("Inverse Document Frequency Diagram")

plt.savefig("inverse_document_frequency.svg")