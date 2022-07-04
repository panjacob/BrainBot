"""
The neural network's training.
Tensorboard tutorial: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
"""

from brainset import *
from model import *
from torch.utils.tensorboard import SummaryWriter

single_batch_test = False
save_model = True
save_dir_path = "models"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# TODO: Comment on this function
def label_to_human_form(labels):
    result = []
    for x in labels:
        result.append(int(x))
    return result


# TODO: Comment on this function
def accuracy_human(a, b):
    result = 0
    for x, y in zip(label_to_human_form(a), label_to_human_form(b)):
        result += 1 if x == y else 0
    return result / len(a)


def main():
    writer = SummaryWriter()
    torch.set_default_dtype(torch.float32)
    brainloader, testloader = load_data()
    device = torch.device("cuda")
    model = OneDNetScaled()
    criterion = nn.BCELoss()  # binary cross entropy
    optimalizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    model.to(device)

    if single_batch_test is True:
        # Preform Single Batch Test
        brainloader = [next(iter(brainloader))]
        print("Single Batch Test Chosen")

    for epoch in range(1001):
        print('epoch', epoch)

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
                writer.add_scalar("Loss/train", loss, epoch)
                loss.backward()
                # clip_grad_norm_(model.parameters(), max_norm=1)
                optimalizer.step()
                preds = [0 if out < 0.5 else 1 for out in outputs]
                acc = accuracy_human(labels, preds)
                train_accuracy.append(acc)
                writer.add_scalar("Accuracy/train", acc, epoch)
                train_loses.append(loss)

        print('accuracy', sum(train_accuracy) / len(train_accuracy))
        print('loss', (sum(train_loses) / len(train_loses)).item())

        model.eval()
        if not epoch % 10 and (epoch or single_batch_test):
            print("Testing")
            accuracy = []
            loses = []
            model.eval()
            with torch.no_grad():
                for inputs, labels, filenames in testloader:
                    inputs = torch.autograd.Variable(inputs.to(device, non_blocking=True))
                    labels = torch.autograd.Variable(labels.to(device, non_blocking=True))
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = [0 if out < 0.5 else 1 for out in outputs]
                    acc = accuracy_human(labels, preds)
                    accuracy.append(acc)
                    loses.append(loss)
                    writer.add_scalar("Loss/test", loss, epoch)
                    writer.add_scalar("Accuracy/test", acc, epoch)
                print('test accuracy', sum(accuracy) / len(accuracy))
                print('test loss', (sum(loses) / len(loses)).item())
                if save_model:
                    save_param = f"E:{epoch}_A:{sum(accuracy) / len(accuracy)}"
                    model.saveModel(save_dir_path,save_param)
                    print("Model Saved")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
