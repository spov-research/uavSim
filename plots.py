import json

import pandas as pd
import seaborn as sns

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

with open("data.json") as file:
    data = json.load(file)

runs = {
    "ddqn": "20221022_140239_dqn_2x",
    "masked": "20221017_121538_masked_cpp",
    "fddqn": "20221022_173831_dqfn_2x",
    "fddqn_pen": "20221025_190358_dqfn_with_penalties"
}

results = {}

metrics = ["cral", "cr", "landed"]

for name, run in runs.items():
    results[name] = {}
    for metric in metrics:
        d = data[run][metric].items()
        steps = []
        values = []
        for step, value in d:
            if int(step) > 1e6:
                break
            steps.append(int(step) / 1000)
            values.append(value)

        results[name].update({"steps": steps, metric: values})

full_dict = {"steps": results["ddqn"]["steps"]}
full_dict.update({f"{method}_{metric}": results[method][metric] for method in runs.keys() for metric in metrics})
original = pd.DataFrame.from_dict(full_dict)

# df["rounded_steps"] = (df["steps"] / 20).astype(int) * 20
df = original.rolling(40, min_periods=10).mean()


plt.figure()
sns.lineplot(data=df, x="steps", y="ddqn_cral", label="DDQN")
sns.lineplot(data=df, x="steps", y="masked_cral", label="Masked-DDQN")
sns.lineplot(data=df, x="steps", y="fddqn_cral", label="$\Phi$-DDQN")
sns.lineplot(data=df, x="steps", y="fddqn_pen_cral", label="$\Phi$-DDQN (pen)")

plt.ylabel("Coverage ratio and landed")
plt.xlabel("Training time [k steps]")
plt.grid()
plt.xlim([0, 500])
plt.ylim([0, 1])

plt.savefig("cral.png", dpi=300, bbox_inches='tight')
# plt.show()
plt.close()

plt.figure()
sns.lineplot(data=df, x="steps", y="ddqn_cr", label="DDQN")
sns.lineplot(data=df, x="steps", y="masked_cr", label="Masked-DDQN")
sns.lineplot(data=df, x="steps", y="fddqn_cr", label="$\Phi$-DDQN")
sns.lineplot(data=df, x="steps", y="fddqn_pen_cr", label="$\Phi$-DDQN (pen)")

plt.ylabel("Coverage ratio")
plt.xlabel("Training time [k steps]")
plt.grid()
plt.xlim([0, 500])
plt.ylim([0, 1])

plt.savefig("cr.png", dpi=300, bbox_inches='tight')
# plt.show()
plt.close()


df = original.rolling(100, min_periods=10).mean()
plt.figure()
sns.lineplot(data=df, x="steps", y="ddqn_landed", label="DDQN")
sns.lineplot(data=df, x="steps", y="masked_landed", label="Masked-DDQN")
sns.lineplot(data=df, x="steps", y="fddqn_landed", label="$\Phi$-DDQN")
sns.lineplot(data=df, x="steps", y="fddqn_pen_landed", label="$\Phi$-DDQN (pen)")

plt.ylabel("Landed")
plt.xlabel("Training time [k steps]")
plt.grid()
plt.xlim([0, 500])
plt.ylim([0, 1.1])

plt.savefig("landed.png", dpi=300, bbox_inches='tight')
# plt.show()
plt.close()
