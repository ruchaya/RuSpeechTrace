from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

def save_confusion(y_true,y_pred,path,normalize="true"):
    labels=sorted(set(map(str,y_true))|set(map(str,y_pred))); cm=confusion_matrix(y_true,y_pred,labels=labels,normalize=normalize)
    fig,ax=plt.subplots(figsize=(max(8,len(labels)*.28),max(7,len(labels)*.28)))
    ConfusionMatrixDisplay(cm,display_labels=labels).plot(ax=ax,xticks_rotation=90,include_values=False)
    fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=250,bbox_inches="tight"); plt.close(fig)

def save_score_distribution(known_scores,unknown_scores,path):
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.hist(known_scores,bins=60,density=True,alpha=.55,label="Known"); ax.hist(unknown_scores,bins=60,density=True,alpha=.55,label="Unknown")
    ax.set_xlabel("ID-oriented score"); ax.set_ylabel("Density"); ax.legend(); fig.tight_layout()
    Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=250,bbox_inches="tight"); plt.close(fig)

def save_heatmap(df,path,title=None):
    fig,ax=plt.subplots(figsize=(max(8,df.shape[1]*.55),max(5,df.shape[0]*.3))); im=ax.imshow(df.to_numpy(float),aspect="auto")
    ax.set_xticks(np.arange(df.shape[1]),labels=df.columns,rotation=90); ax.set_yticks(np.arange(df.shape[0]),labels=df.index)
    if title: ax.set_title(title)
    fig.colorbar(im,ax=ax); fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=250,bbox_inches="tight"); plt.close(fig)
