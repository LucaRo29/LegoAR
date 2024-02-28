import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
import csv

import pandas as pd


# Example usage
folder_path = 'runs'
scalar_name = 'Val/val_top5_acc'  # Replace with your desired scalar name


def main():

    plot_scalar_from_events(folder_path, scalar_name)

    
    # matrix = create_matrix(read_csv_file('top5real.csv'))
    # createCM_fig(np.array(matrix))

    # top1average('top5test.csv')

    

def plot_scalar_from_events(folder_path, scalar_name):
    event_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    scalar_data = {}  # Dictionary to store scalar values

    
    for event_folder in event_folders:
        event_folder_path = os.path.join(folder_path, event_folder)
       
        event_files = [f for f in os.listdir(event_folder_path) if f.startswith('events')]

        for event_file in event_files:
            event_path = os.path.join(event_folder_path, event_file)
            event_acc = EventAccumulator(event_path)
            event_acc.Reload()
            
            
            if scalar_name not in event_acc.Tags()['scalars']:
                continue

            scalar_events = event_acc.Scalars(scalar_name)
            if scalar_events:
                # Extract scalar values and steps
                steps = [event.step for event in scalar_events]
                values = [event.value for event in scalar_events]

                scalar_data[event_folder + '_' + event_file] = (steps, values)

        if not scalar_data:
            print(f"No events found for scalar '{scalar_name}' in the specified folder.")
            continue
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title('Validation Top5-Accuracy Curve',fontsize=26)
    plt.xlabel('Epochs',fontsize=20)
    plt.ylabel('Top5-Accuracy',fontsize=20)

    for event_file, (steps, values) in scalar_data.items():
        
        label = event_file.split('_')[0]+'_'+event_file.split('_')[1]
        plt.plot(steps, values, label=label, linewidth=1)

    plt.legend()
    #plt.savefig(f'{scalar_name}.png')
    plt.savefig('valacc5.pdf')

def top1average(csv_file):


    df = pd.read_csv(csv_file,header=None)

    
    list_of_lists = [[] for _ in range(132)]
    
    for _ , row in df.iterrows():
                      
        list_of_lists[row[0]].append(row[1])
        
    metrics = pd.DataFrame(columns = ['mean','std'])
    
    

    
    sum = 0
    for l in list_of_lists:
        l.sort()
        

    list_of_lists = [inner_list[2:-2] for inner_list in list_of_lists]
    
    with open('testpred.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(list_of_lists)


    count = 0


    for l in list_of_lists:
        
        new_row = pd.DataFrame([{"mean": np.mean(np.array(l)),  "std": np.std(np.array(l))}])
        metrics = pd.concat([metrics,new_row],ignore_index=True)
        


    plot_mean_std(metrics)
    print(metrics.head)


def read_csv_file(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = list(reader)
    return data

def create_matrix(data):
    matrix = [[0 for i in range(132)] for j in range(132)]
    for row in data:
        column_index = int(row[0])
        for value in row[1:]:
            matrix[  column_index][  int(value)] += 1
            
            if int(value) == column_index:
                
                break


    with open('top5CM.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(matrix)           
    return matrix

def createCM_fig(confusion_matrix):
    # create a figure and axes object
    fig, ax = plt.subplots(figsize=(30, 30))

    # set the color map
    cmap = plt.get_cmap('Blues')

    # create a heatmap of the confusion matrix
    im = ax.imshow(confusion_matrix, cmap=cmap, interpolation='nearest')

    # set the ticks and labels for the x-axis and y-axis
    ax.set_xticks(np.arange(confusion_matrix.shape[1]))
    ax.set_yticks(np.arange(confusion_matrix.shape[0]))
    ax.set_xticklabels(['step {}'.format(i) for i in range(confusion_matrix.shape[1])])
    ax.set_yticklabels(['step {}'.format(i) for i in range(confusion_matrix.shape[0])])
    ax.set_xlabel("Predictions", fontsize=70)
    ax.set_ylabel("Ground Truth", fontsize=70)

    # set the rotation angle of the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=67, ha="right",
             rotation_mode="anchor")

    # loop over data dimensions and create text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            if confusion_matrix[i, j] == 0:
                continue
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="black", fontsize=6)

    # add a colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # set the title of the figure
    ax.set_title("Confusion Matrix", fontsize=100)

    plt.savefig('Top5CMr.png')

def plot_mean_std(df):
   
    fig, ax = plt.subplots(figsize=(24, 5))

    x_values = range(len(df))
    for idx, row in df.iterrows():
        mean = row['mean'] - idx
        std = row['std']
        ax.errorbar(idx, mean, yerr=std, fmt='o',markersize=9) 

    ax.plot([0,131], [0,0], linestyle='-', color='red', label=' Perfect Accuracy',linewidth=1)

    ax.set_xticks(x_values)
    ax.set_xticklabels(df.index)
    #ax.set_xlabel('Ground Truth',fontsize=20)
    #ax.set_ylabel('Mean Predictions +- Std',fontsize=26, rotation=-90)
    plt.yticks(fontsize=20)
    ax.set_title('Error Spread on the Test Dataset without Outliers',fontsize=30) #  without Outliers
   
    ax.set_ylim(-16,14)
    ax.set_xlim(-1,132)

    
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="center",
             fontsize=15)

    plt.tight_layout()
    plt.savefig('errortesttrim.pdf')


if __name__ == '__main__':
    main()
