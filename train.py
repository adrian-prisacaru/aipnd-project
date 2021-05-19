import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from utils import build_model, determine_device
from pathlib import Path

# transforms used for training

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
# transforms used for validation and testing
data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


def get_train_loader(data_dir):
    train_dir = data_dir + '/train'
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    return train_loader, train_data


def get_validation_loader(data_dir):
    valid_dir = data_dir + '/valid'
    validation_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
    return torch.utils.data.DataLoader(validation_data, batch_size=64)


def get_test_loader(data_dir):
    test_dir = data_dir + '/test'
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)
    return torch.utils.data.DataLoader(test_data, batch_size=64)


def train(model, device, criterion, optimizer, train_loader, validation_loader, epochs):
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(validation_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validation_loader):.3f}")
                running_loss = 0
                model.train()
    print("Training done")


def test_model(model, device, criterion, test_loader):
    print("\nLoad test set data...")
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(test_loader):.3f}.. "
          f"Test accuracy: {accuracy/len(test_loader):.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint")
    parser.add_argument('data_dir', help='Data folder')
    parser.add_argument('--save_dir', default='checkpoints', help='Folder to save checkpoints')
    parser.add_argument('--arch', default='vgg16', help='Model Architecture', choices=['vgg13', 'vgg16', 'vgg19'])
    parser.add_argument('--learning_rate', default=0.003, type=float, help='Learning rate')
    parser.add_argument('--hidden_units', default=[4096, 1000], nargs="+", type=int, help='Hidden units')
    parser.add_argument('--epochs', default=3, type=int, help='Epochs')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU training')
    return parser.parse_args()


def main():
    args = parse_args()
    device = determine_device(args.gpu)

    # load training and validation data
    train_loader, train_data = get_train_loader(args.data_dir)
    validation_loader = get_validation_loader(args.data_dir)

    # build model and define criterion and optimizer
    output = len(train_data.class_to_idx)
    model = build_model(args.arch, args.hidden_units, output, args.dropout)
    criterion = nn.NLLLoss()
    # should lr match the model learning rate?
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    model.to(device)

    # train
    train(model, device, criterion, optimizer, train_loader, validation_loader, args.epochs)

    # test
    test_loader = get_test_loader(args.data_dir)
    test_model(model, device, criterion, test_loader)

    # save checkpoint
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'optimizer': optimizer.state_dict(),
        'epochs': args.epochs,
        'arch': args.arch,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'hidden_units': args.hidden_units
    }
    # create the file if it doesn't exist
    checkpoint_file = Path("{}/checkpoint.pth".format(args.save_dir))
    checkpoint_file.touch(exist_ok=True)
    torch.save(checkpoint, checkpoint_file)

    print("\n Checkpoint saved successfully")


if __name__ == "__main__":
    main()
