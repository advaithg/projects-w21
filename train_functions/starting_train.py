import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
from data_augment import DataAugmenter
import matplotlib.pyplot as plt

def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path, device, usePretrained
):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]
    weight_decay = hyperparameters["weight_decay"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters(), weight_decay = weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    augment = False

    model.to(device)

    # Make true to train from preexisting net
    if(usePretrained):
        checkpoint = torch.load(f"models/model.pt", map_location=device)
        # initialize state_dict from checkpoint to model
        model.load_state_dict(checkpoint['state_dict'])
        # initialize optimizer from checkpoint to optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        augment = True
    
    augment = True
    
    loss_fn.to(device)

    # Initialize summary writer (for logging)
    writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    # Initialize data augmenter
    augmenter = DataAugmenter()

    cumul_loss = 0
    step = 0
    n_correct = 0
    n_total = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        
        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")

            # TODO: Backpropagation and gradient descent

            img, labels = batch
            
            # Make true to use data augmenter
            if(augment):
                img = augmenter.applyAugmentations(img)
                #image = img.int()
                #plt.imshow(image[0].permute(2, 1, 0))
                #plt.show()

            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()
            
            predictions = model.forward(img)

            loss = loss_fn(predictions, labels)

            n_correct += (predictions.argmax(axis=1) == labels).sum().item()
            n_total += len(predictions.argmax(axis=1))
            cumul_loss += loss.item()

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                accuracy = n_correct / n_total
                print(f"Training accuracy: {accuracy * 100}")

                # Log the results to Tensorboard.
                writer.add_scalar("train_loss", cumul_loss / n_eval, global_step = step * 100 / n_eval)
                writer.add_scalar("train_accuracy", accuracy * 100, global_step = step * 100 / n_eval)

                cumul_loss = 0
                n_correct = 0
                n_total = 0

                # Save the model
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'augment': augment
                }
                
                torch.save(checkpoint, f"models/model.pt")

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                model.eval()
                with torch.no_grad():
                    evaluate(val_loader, model, loss_fn, device, step, writer, n_eval, batch_size)
                
   
            model.train()
            step += 1
            
            loss.backward()
            optimizer.step()
            
        print("Epoch ", epoch, "Loss ", loss.item())
    
    # Save the model
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'augment': augment
    }
    
    torch.save(checkpoint, f"models/model.pt")


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    #print(outputs.shape)
    #print(labels.shape)

    n_correct = (outputs == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, device, step, writer, n_eval, batch_size):
    """
    Computes the loss and accuracy sof a model on the validation dataset.
    """
    model.eval() #eval mode so network doesn't learn from test dataset
    
    n_correct = 0
    n_total = 0
    cumul_loss = 0

    for i, data in enumerate(val_loader):
        input_data, labels = data
        input_data, labels = input_data.to(device), labels.to(device)
        predictions = model.forward(input_data)
        n_correct += (predictions.argmax(axis=1) == labels).sum().item()
        n_total += len(predictions.argmax(axis=1))        
        #accuracy = compute_accuracy(predictions.argmax(axis=1), labels)
        loss = loss_fn(predictions, labels)
        cumul_loss += loss.item()
        
    print(f"Validation Accuracy: {n_correct/n_total * 100} Loss: {loss}")
    writer.add_scalar("validation_loss", cumul_loss * batch_size / n_total, global_step = step * 100 / n_eval)
    writer.add_scalar("validation_accuracy", n_correct * 100 / n_total, global_step = step * 100 / n_eval)