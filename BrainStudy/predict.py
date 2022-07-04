#Predict using CNNmodel


from brainset import *
from model import *
from train import *

load_model_path = "models/Model_OneDNetScaled_04.07.2022_13:11_E:11_A:0.5406126155878468.pt"
test_batch_size = 1


def main():

    torch.set_default_dtype(torch.float32)
    _, testloader = load_data(test_batch_size=test_batch_size)
    device = torch.device("cuda")
    model = OneDNetScaled()
    model.loadModel(load_model_path,device)
    criterion = nn.BCELoss()  # binary cross entropy
    model.to(device)
    model.eval()

    print("This is prediction script")
    print("Batch size is ", test_batch_size)

    with torch.no_grad():
        print("Predictions:")
        accuracy = []
        loses = []

        user_in = True
        while user_in:
            inputs, labels, filenames = next(iter(testloader))
            print("New Prediction: ")
            inputs = torch.autograd.Variable(inputs.to(device, non_blocking=True))
            labels = torch.autograd.Variable(labels.to(device, non_blocking=True))
            outputs = model(inputs)
            preds = [0 if out < 0.5 else 1 for out in outputs]
            loss = criterion(outputs, labels)
            acc = accuracy_human(labels, preds)
            accuracy.append(acc)
            loses.append(loss)
            print("Predicted : ", preds, " with ", np.array(outputs.cpu()))
            print("Labels : ", np.array(labels.cpu()))
            print("Acc = ",acc)
            print("Loss = ",loss.item())
            uin = input("Continue?\n")
            if uin == 'n':
                user_in = False

        print('Mean accuracy', sum(accuracy) / len(accuracy))
        print('Mean loss', (sum(loses) / len(loses)).item())



if __name__ == "__main__":
    main()



