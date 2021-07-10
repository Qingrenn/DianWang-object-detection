import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    pred_result_path = "./pred_results3"
    file_list = os.listdir(pred_result_path)
    scores = {"bage": [], 'offground': [], 'ground': [], 'safebelt': []}
    num2label = {1:'bage', 2:'offground', 3:'ground', 4:'safebelt'}

    for file in file_list:
        file_path = os.path.join(pred_result_path, file.strip())
        img_info = {}
        with open(file_path) as f:
            img_info  = json.load(f)
            for obj in img_info['objs']:
                label = int(obj['label'])
                scores[num2label[label]] += [round(obj['score'], 2)]
    
    for label, score in scores.items():
        draw(label, score)
    
def draw(label, score):
    fig, ax = plt.subplots(1,1)
    bins = list(np.arange(0, 1.1, 0.05))
    ax.hist(score, bins=bins, histtype="stepfilled", alpha=0.6)
    # ax.set_ylabel('Number')
    # ax2 = ax.twinx()
    # sns.kdeplot(score, cumulative=True, shade=True)
    fig.savefig('./' + label  + '_scores.png')
    
    

    

if __name__ == "__main__":
    main()