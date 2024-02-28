import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

from torch.utils.data import SubsetRandomSampler
from tqdm.auto import tqdm
from model import build_model
from datasets import get_datasets, get_data_loaders, get_full_dataset, get_real_test_set

from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision.utils import save_image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torchvision.models as models
# from pandas import DataFrame
# import seaborn as sn
from torchvision import transforms
from sklearn.model_selection import KFold
import configparser
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os 
import csv

np.set_printoptions(precision=None, threshold=100000, edgeitems=None, linewidth=1000)

torch.cuda.manual_seed_all(0)
torch.manual_seed(0)

config = configparser.ConfigParser()
config.read('config.ini')

DEVICE = config.get('TRAINING', 'DEVICE')
epochs = config.getint('TRAINING', 'epochs')
kfold = config.getboolean('TRAINING', 'kfold')
batch_size = config.getint('TRAINING', 'batch_size')
num_workers = config.getint('TRAINING', 'num_workers')

Checkpoint_path = config.get('MODEL', 'Checkpoint_path')
pretrained = config.getboolean('MODEL', 'pretrained')
lr = config.getfloat('MODEL', 'lr')

for section in config.sections():
    print(f'[{section}]')
    for key, value in config.items(section):
        print(f'{key} = {value}')


def main():
    tb = SummaryWriter(comment = config.get('MODEL', 'model_load_path'))
    config_summary = '\n'.join([f'{section}:\n{dict(config[section])}' for section in config.sections()])
    tb.add_text('Config', config_summary)

    
    # train_static_split(tb)    
    #val_model(tb,config.get('MODEL', 'model_load_path'))
    test_model_real(tb, config.get('MODEL', 'model_load_path'))

    

def infere_image(model_path, img_path):
    num_classes = 132

    model = build_model(
        pretrained=pretrained,
        fine_tune=True,
        num_classes=num_classes
    ).to(DEVICE)
    # optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss()

    # img_path = 'step_12_20221116_112232_gimp-2.png'
    img = Image.open(img_path)
    img = img.convert('RGB')

    # Define the transformation to be applied to the image
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformation to the image
    img = transform(img)

    # Add a batch dimension to the image
    img = img.unsqueeze(0)

    model = model.to(DEVICE)

    # Move the input tensor to the GPU
    img = img.to(DEVICE)

    model.eval()

    # Make a prediction
    with torch.no_grad():
        output = model(img)

    # Convert the output to probabilities
    probs = torch.softmax(output, dim=1)

    print(probs)

    softmax_bar_plot(probs, img_path)


def test_model_real(tb, model_path, classes=None, num_classes=132):
    dataset_test, dataset_classes, test_loader = get_real_test_set('../data/real_image_test', classes)

    #print(f"[INFO]: Class names: {dataset_classes}\n")

    model = build_model(
        pretrained=pretrained,
        fine_tune=True,
        num_classes=num_classes
    ).to(DEVICE)

    print("\n\n")
    print("#" * 50)
    print("Testing Model on real images")
    print("#" * 50)
    print("\n\n")

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_top5_acc, test_conf_matrix = validate(model, test_loader, criterion, num_classes)
    tb.add_scalar("EVALUATION/Real/Loss", test_loss)
    tb.add_scalar('EVALUATION/Real/Acc', test_acc)
    tb.add_scalar('EVALUATION/Real/test_top5_acc', test_top5_acc)

    print(f"Real Test loss: {test_loss:.3f}, Test acc: {test_acc:.3f}, Test top5 acc: {test_top5_acc:.3f},")
    tb.add_figure("EVALUATION/Real/Confusion Matrix", create_confusion_matrix(test_conf_matrix))


def val_model(tb, model_path):
    dataset_train, dataset_valid, dataset_test, dataset_classes = get_datasets()
    num_classes = len(dataset_classes)
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    #print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, val_loader, test_loader = get_data_loaders(dataset_train, dataset_valid, dataset_test)

    model = build_model(
        pretrained=pretrained,
        fine_tune=True,
        num_classes=num_classes
    ).to(DEVICE)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc, val_top5_acc, val_conf_matrix = validate(model, val_loader, criterion, num_classes)

    tb.add_scalar("EVALUATION/Val/Loss", val_loss)
    tb.add_scalar('EVALUATION/Val/Acc', val_acc)
    tb.add_scalar('EVALUATION/Val/val_top5_acc', val_top5_acc)

    print(f"Validation loss: {val_loss:.3f}, validation acc: {val_acc:.3f}, validation top5 acc: {val_top5_acc:.3f},")

    tb.add_figure("EVALUATION/Val/Confusion Matrix", create_confusion_matrix(val_conf_matrix))

    test_loss, test_acc, test_top5_acc, test_conf_matrix = validate(model, test_loader, criterion, num_classes)
    tb.add_scalar("EVALUATION/Test/Loss", test_loss)
    tb.add_scalar('EVALUATION/Test/Acc', test_acc)
    tb.add_scalar('EVALUATION/Test/test_top5_acc', test_top5_acc)

    print(f"Test loss: {test_loss:.3f}, Test acc: {test_acc:.3f}, Test top5 acc: {test_top5_acc:.3f},")
    tb.add_figure("EVALUATION/Test/Confusion Matrix", create_confusion_matrix(test_conf_matrix))

    


def train_static_split(tb, classes=None):
    print('Using static train/val split')

    dataset_train, dataset_valid, dataset_test, dataset_classes = get_datasets(classes)
    num_classes = len(dataset_classes)
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Number of test images: {len(dataset_test)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, val_loader, _ = get_data_loaders(dataset_train, dataset_valid, dataset_test)
    

    model = build_model(
        pretrained=pretrained,
        fine_tune=True,
        num_classes=num_classes
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, mode='min', patience=5,cooldown= 3, verbose=True)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)

        val_loss, val_acc, val_top5_acc, val_conf_matrix = validate(model, val_loader, criterion, num_classes)

        tb.add_scalar("Train/Loss", train_loss, epoch)
        tb.add_scalar('Train/Acc', train_acc, epoch)
        tb.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Training loss: {train_loss:.3f}, Training acc: {train_acc:.3f}")

        tb.add_scalar("Val/Loss", val_loss, epoch)
        tb.add_scalar('Val/Acc', val_acc, epoch)
        tb.add_scalar('Val/val_top5_acc', val_top5_acc, epoch)

        print(
            f"Validation loss: {val_loss:.3f}, validation acc: {val_acc:.3f}, validation top5 acc: {val_top5_acc:.3f},")
        print('-' * 50)

        scheduler.step(val_loss)

        if (epoch > 0 and (epoch + 1) % 5 == 0):
            save_checkpoint(epoch, model, optimizer, criterion)
            tb.add_figure("Confusion Matrix", create_confusion_matrix(val_conf_matrix), epoch)

    print('TRAINING COMPLETE')
    save_checkpoint(epoch, model, optimizer, criterion)
    # tb.add_figure("Confusion Matrix", create_confusion_matrix(conf_matrix), epoch)


# Training function.
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        
        # # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.sampler))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, loader, criterion, num_classes):
    model.eval()
    print('Validation')
    val_running_loss = 0.0
    val_running_correct = 0
    val_top5_correct = 0

    conf_matrix = torch.zeros(num_classes, num_classes)
    top5_matrix = []
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            

            image, labels = data
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == labels).sum().item()

           

            # Calculate the top-5 accuracy.
            if (num_classes > 5):
                _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
                for j in range(labels.size(0)):

                    val_top5_correct += labels[j] in top5_preds[j]
                    helper= []
                    helper.append(labels[j].cpu().numpy().item())
                    for x in range(0,5):
                        
                        helper.append(top5_preds[j].cpu().numpy()[x])

                    top5_matrix.append(helper)
                    
            else:
                val_top5_correct = val_running_correct

            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

            counter += 1

    # Loss and accuracy for the complete epoch.
    epoch_loss = val_running_loss / counter
    epoch_acc = 100. * (val_running_correct / len(loader.sampler))
    epoch_top5_acc = 100. * (val_top5_correct / len(loader.sampler))

    # print(conf_matrix)
    # print(top5_matrix)

    with open('top5.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(top5_matrix)
        writer.writerow([0,0,0,0,0])

    return epoch_loss, epoch_acc, epoch_top5_acc, conf_matrix


def save_checkpoint(epoch, model, optimizer, loss):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'checkpoints/model_checkpoint{epoch}.pth')


def create_confusion_matrix(confusion_matrix):
    """
    Plot a confusion matrix.

    Args:
        confusion_matrix (torch.Tensor): a tensor containing the confusion matrix

    Returns:
        fig (matplotlib.figure.Figure): a figure object containing the confusion matrix plot
    """

    # convert the confusion matrix tensor to a numpy array
    confusion_matrix = confusion_matrix.numpy().astype(int)

    # create a figure and axes object
    fig, ax = plt.subplots(figsize=(25, 25))

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

    return fig


def softmax_bar_plot(softmax_tensor, name):
    """
    Creates a bar plot from a tensor of softmax probabilities and saves it to a file.

    Parameters:
    softmax_tensor (numpy.ndarray): A 1D numpy array of softmax probabilities for each class.
    output_path (str): The path to save the output image.

    Returns:
    None

    """
    softmax_tensor = softmax_tensor.cpu().squeeze()
    print(softmax_tensor.shape)
    # Create a list of class labels
    num_classes = len(softmax_tensor)
    class_labels = [f"Class {i + 1}" for i in range(num_classes)]

    # Create a bar plot of the softmax probabilities
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.bar(class_labels, softmax_tensor)
    ax.set_title("Softmax Probabilities")
    ax.set_xlabel("Class")
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    plt.setp(ax.get_xticklabels(), rotation=67, ha="right",
             rotation_mode="anchor")

    # Save the plot to a file
    fig.savefig(f"Softmax_bar_{name}.png")
    plt.close(fig)

def remove_zero_rows_and_columns_keep_indices(confusion_matrix):
  """Removes all rows and columns which only consist of zeros from a confusion matrix.

  Args:
    confusion_matrix: A confusion matrix.

  Returns:
    A new confusion matrix with the rows and columns removed, and the indices
    of the remaining rows and columns.
  """

  # Create a boolean mask to indicate which rows and columns contain any non-zero values.
  rows_mask = confusion_matrix.any(axis=1)
  columns_mask = confusion_matrix.any(axis=0)

  # Get the indices of the remaining rows and columns.
  rows_indices = np.nonzero(rows_mask)[0]
  columns_indices = np.nonzero(columns_mask)[0]

  # Create a new confusion matrix with the rows and columns removed.
  new_confusion_matrix = confusion_matrix[rows_indices, columns_indices]

  # Return the new confusion matrix and the indices of the remaining rows and columns.
  return new_confusion_matrix, rows_indices, columns_indices

# def writeCMtoFile(cm,rows_indices=size(cm), columns_indices=size(cm)):
#     with open("confusion_matrix.csv", "w") as csvfile:
#   writer = csv.writer(csvfile, delimiter=",")
#   for row_index, row in enumerate(new_confusion_matrix):
#     writer.writerow(row)

if __name__ == '__main__':
    main()


def train_kfold(tb):
    print('Using kfold validation strategy')

    dataset = get_full_dataset()
    num_classes = len(dataset.classes)
    print(f"[INFO]: Class names: {dataset.class_to_idx}")
    print(f"[INFO]: Dataset len {len(dataset)}")

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = kf.split(dataset)

    fold_counter = 0
    for fold, (train_idx, val_idx) in enumerate(folds):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers
        )

        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers
        )

        model = build_model(
            pretrained=pretrained,
            fine_tune=True,
            num_classes=num_classes
        ).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        criterion = nn.CrossEntropyLoss()

        print('#' * 50)
        print(f"Fold {fold}")
        print('#' * 50)
        for epoch in range(epochs):
            # Set your model to training mode
            print("")
            print(f"[INFO]: Epoch {epoch + 1} of {epochs}")

            train_loss, train_acc = train(model, train_loader, optimizer, criterion)
            val_loss, val_acc, conf_matrix = validate(model, val_loader, criterion, num_classes)

            scheduler.step(val_loss)

            print(f"Training loss: {train_loss:.3f}, training acc: {train_acc:.3f}")
            print(f"Validation loss: {val_loss:.3f}, validation acc: {val_acc:.3f}")
            print('-' * 50)
            tb.add_scalar(f"kfold/fold{fold}/Train/Loss", train_loss, epoch)
            tb.add_scalar(f"kfold/fold{fold}/Val/Loss", val_loss, epoch)
            tb.add_scalar(f"kfold/fold{fold}/Train/Acc", train_acc, epoch)
            tb.add_scalar(f"kfold/fold{fold}/Val/Acc", val_acc, epoch)

            print(conf_matrix)
            tb.add_figure(f"kfold/fold{fold}/Confusion Matrix", create_confusion_matrix(conf_matrix), epoch)
            
        print(f'TRAINING COMPLETE FOLD {fold}')
        fold_counter += 1
        if fold_counter == 2:
            break

    