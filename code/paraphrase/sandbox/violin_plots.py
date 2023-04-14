import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")
ax = sns.violinplot(x="day", y="total_bill", data=tips)
ax = sns.swarmplot(x="day", y="total_bill", data=tips, color=".25")

#plt.show()
plt.savefig(f'violinplot-points.png')


plt.clf()
ax = sns.violinplot(x="day", y="total_bill", data=tips, hue='smoker')
plt.savefig(f'violinplot-dual.png')
