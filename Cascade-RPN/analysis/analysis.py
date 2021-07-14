import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os.path as osp
import os

def draw_loss(df):
    df = df.loc[df['loss']!=0]
    if df['mode'].iloc[0] == 'val':
        losses = ('s0.loss_rpn_reg', 's1.loss_rpn_cls', 's1.loss_rpn_reg', 'loss_cls', 'loss_bbox', 'loss')
        plt.figure(figsize=(8,6))
        for idx, loss in enumerate(losses):
            ax = plt.subplot(3, 2, idx+1)
            df = df.loc[df[loss]!=0]
            ax.plot(df['epoch'], df[loss])
            ax.set_title(loss)
        plt.subplots_adjust(hspace=0.5)
    else:
        df.plot(y='loss')
    plt.savefig(osp.join(output_dir, df['mode'].iloc[0] + '_loss.png'))

def draw_map(df):
    assert df['mode'].iloc[0] == 'val'
    maps = ("bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l")
    plt.figure(figsize=(8,6))
    for idx, map in enumerate(maps):
        ax = plt.subplot(3, 2, idx+1)
        df = df.loc[df[map]!=0]
        ax.plot(df['epoch'], df[map])
        ax.set_title(map)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(osp.join(output_dir, 'val_mAP.png'))

def draw_acc(df):
    assert df['mode'].iloc[0] == 'val'
    df = df.loc[df['acc']!=0]
    df.plot(x='epoch', y='acc')
    plt.savefig(osp.join(output_dir, 'val_acc.png'))

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--json_path', help='output json log', default='../WorkHome/20210710_232308.log.json')
    args = parser.parse_args()
    
    global output_dir
    output_dir = (args.json_path).split('/')[-1].split('.')[0] + '_analysis'
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.DataFrame()
    with open(args.json_path, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            df = df.append(eval(line.strip()),ignore_index=True)
    df = df.fillna(0)
    train_df = df.loc[df['mode']=='train']
    val_df = df.loc[df['mode']=='val']
    
    draw_loss(val_df)
    draw_loss(train_df)

    draw_map(val_df)
    draw_acc(val_df)

    