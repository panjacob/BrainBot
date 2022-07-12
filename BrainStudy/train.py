"""
The neural network's training.
Tensorboard tutorial: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
using one of those commands : tensorboard --logdir=BrainStudy/runs or  tensorboard --logdir=runs
"""
import time
from brainset import *
from model import *
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import seaborn as sn
from sklearn.metrics import roc_auc_score, RocCurveDisplay

load_pickled_data = True
single_batch_test = False
save_model = True
save_dir_path = "models"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Converts model output (the number form 0. to 1.) into binary classification ( 0 or 1 )
def predictLabels(model_outputs):
    threshold = 0.5 # decides where to differentiate 1 form 0
    return torch.where(model_outputs >= threshold,1.,0.)


# Calculates accuracy of the binary model
def accuracyHuman(labels, model_outputs):
    predictions = predictLabels(model_outputs)
    return torch.sum(torch.eq(labels,predictions)) / len(predictions)


class BinaryConfusionMatrix:
    def __init__(self):
        self.true_positives = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.true_negatives = 0

    def append(self,labels, model_outputs):
        predictions = predictLabels(model_outputs)
        confusion_vector = predictions / labels
        self.true_positives = self.true_positives + torch.sum(confusion_vector == 1).item()
        self.false_negatives = self.false_negatives + torch.sum(confusion_vector == 0).item()
        self.false_positives = self.false_positives + torch.sum(confusion_vector == float('inf')).item()
        self.true_negatives = self.true_negatives + torch.sum(torch.isnan(confusion_vector)).item()

    def writeToTensorboard(self):
        cols = ['Predicted Positive', 'Predicted Negative']
        rows = ["Actual Positive", "Actual Negative"]
        matrix_list = [[self.true_positives, self.false_negatives], [self.false_positives, self.true_negatives]]
        df_cm = pd.DataFrame(matrix_list, cols, rows)
        return sn.heatmap(df_cm, annot=True, fmt='d').get_figure()

'''
def createRocFig(y_true_all, prob_pred_all):
    roc_auc = roc_auc_score(y_true_all, prob_pred_all, average=None, multi_class='ovr')
    roc_auc_avg = roc_auc_score(y_true_all, prob_pred_all, average='macro', multi_class='ovr')
    y_true_binary = [true > 0 for true in y_true_all]
    y_pred_binary = [pred >= 0.5 for pred in prob_pred_all]
    rocs = [RocCurveDisplay.from_predictions(y_true_binary[i], y_pred_binary[i]) for i in range(2)]
    rocs_x = [roc.line_.get_xdata() for roc in rocs]
    rocs_y = [roc.line_.get_ydata() for roc in rocs]
    styles = ['-r', '-g', '-b', '-c', '-m', '-y']
    roc_fig, roc_axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    roc_axs = roc_axs.flatten()
    for i in [0, 2, 8]:
        roc_axs[i].axis('off')
    class_names = ['Mental','Off']
    for i in range(2):
        roc_axs[1].plot(rocs_x[i], rocs_y[i], styles[i], label=f"{class_names[i]} - AUC={roc_auc[i]:.3f}")
        if i != 0:
            roc_ax = roc_axs[i + 2]
            roc_ax.plot(rocs_x[i], rocs_y[i], styles[i], label=f"{class_names[i]} - AUC={roc_auc[i]:.3f}")
            roc_ax.legend(loc='lower right')
    roc_axs[1].legend(loc='best', fontsize='x-small')
    return roc_auc_avg, roc_fig

    conf_matrix_heatmap = sn.heatmap(df_cm, annot=True).get_figure()
    roc_auc = roc_auc_score(y_true_all, prob_pred_all, average=None, multi_class='ovr')
    roc_auc_avg = roc_auc_score(y_true_all, prob_pred_all, average='macro', multi_class='ovr')
    y_true_binary = [[true == i for true in y_true_all] for i in range(classes_count)]
    y_pred_binary = [[probs[i] for probs in prob_pred_all] for i in range(classes_count)]
    rocs = [RocCurveDisplay.from_predictions(y_true_binary[i], y_pred_binary[i]) for i in range(classes_count)]
    rocs_x = [roc.line_.get_xdata() for roc in rocs]
    rocs_y = [roc.line_.get_ydata() for roc in rocs]
    styles = ['-r', '-g', '-b', '-c', '-m', '-y']
    roc_fig, roc_axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    roc_axs = roc_axs.flatten()
    for i in [0, 2, 8]:
        roc_axs[i].axis('off')
    for i in range(classes_count):
        roc_axs[1].plot(rocs_x[i], rocs_y[i], styles[i], label=f"{class_names[i]} - AUC={roc_auc[i]:.3f}")
        if i != 0:
            roc_ax = roc_axs[i + 2]
            roc_ax.plot(rocs_x[i], rocs_y[i], styles[i], label=f"{class_names[i]} - AUC={roc_auc[i]:.3f}")
            roc_ax.legend(loc='lower right')
    roc_axs[1].legend(loc='best', fontsize='x-small')
'''

def main():
    # Measure training time
    start_time = time.perf_counter()
    print("Training Experiment")
    writer = SummaryWriter()
    torch.set_default_dtype(torch.float32)
    brainloader, testloader = load_data(load_pickled_data=load_pickled_data)
    #brainloader.dataset.stats()
    print("Data Loaded")
    device = torch.device("cuda")
    model = OneDNetScaled()
    criterion = nn.BCELoss()  # binary cross entropy
    optimalizer = torch.optim.Adam(model.parameters(), lr=0.000005)
    model.to(device)

    if single_batch_test is True:
        # Preform Single Batch Test
        brainloader = [next(iter(brainloader))]
        print("Single Batch Test Chosen")

    print("Training Model...")
    for epoch in range(1001):
        print('epoch', epoch)
        epoch_time = time.perf_counter() # Measure one epoch time



        train_accuracy = []
        train_loses = []
        model.train()
        for inputs, labels, filenames in brainloader:
            inputs = torch.autograd.Variable(inputs.to(device, non_blocking=True))
            labels = torch.autograd.Variable(labels.to(device, non_blocking=True))

            with torch.set_grad_enabled(True):
                optimalizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # loss = 100 - 2*abs(loss - 50)

                loss.backward()
                # clip_grad_norm_(model.parameters(), max_norm=1)
                optimalizer.step()
                acc = accuracyHuman(labels, outputs)
                train_accuracy.append(acc.item())

                train_loses.append(loss)

        writer.add_scalar("Loss/train", (sum(train_loses) / len(train_loses)).item(), epoch)
        writer.add_scalar("Accuracy/train", sum(train_accuracy) / len(train_accuracy), epoch)
        print('accuracy', sum(train_accuracy) / len(train_accuracy))
        print('loss', (sum(train_loses) / len(train_loses)).item())
        print('time',time.perf_counter()-epoch_time)


        model.eval()
        if not epoch % 1 and (epoch or single_batch_test):
            print("Testing")
            eval_time = time.perf_counter()
            accuracy = []
            loses = []
            conf_matrix = BinaryConfusionMatrix()
            labels_list = []
            outputs_list = []
            model.eval()
            with torch.no_grad():
                for inputs, labels, filenames in testloader:
                    inputs = torch.autograd.Variable(inputs.to(device, non_blocking=True))
                    labels = torch.autograd.Variable(labels.to(device, non_blocking=True))
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    acc = accuracyHuman(labels, outputs)
                    accuracy.append(acc.item())
                    loses.append(loss)
                    conf_matrix.append(labels,outputs)
                    labels_list.extend(labels.cpu())
                    outputs_list.extend(outputs.cpu())
                writer.add_scalar("Loss/test", (sum(loses) / len(loses)).item(), epoch)
                writer.add_scalar("Accuracy/test", sum(accuracy) / len(accuracy), epoch)
                writer.add_figure("ConfusionMatrix/test", conf_matrix.writeToTensorboard(), epoch)
                #roc_auc_avg, roc_fig = createRocFig(labels_list, outputs_list)
                #writer.add_scalar("Average AUC-ROC", roc_auc_avg, epoch)
                #writer.add_figure("ROC", roc_fig, epoch)
                plt.close('all')
                print('test accuracy', sum(accuracy) / len(accuracy))
                print('test loss', (sum(loses) / len(loses)).item())
                print('test time', time.perf_counter()  - eval_time)
                if not epoch % 10 and save_model:
                    save_param = f"E.{epoch}_A.{sum(accuracy) / len(accuracy)}"
                    model.saveModel(save_dir_path,save_param)
                    print("Model Saved")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
