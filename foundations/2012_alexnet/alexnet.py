import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from timeit import default_timer as timer


class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)
    
def train_one_epoch(
    epoch_index,
    training_dataloader,
    model, loss_fn,
    optimizer,
    writer
):
    total_loss = 0.0
    running_loss = 0.0
    total_top1_error = 0.0
    total_top5_error = 0.0
    running_top1_error = 0.0
    running_top5_error = 0.0
    
    # Ensure model is in training mode
    model.train()
    
    batch = 0
    for imgs, classes in tqdm(training_dataloader, desc='Training'):
        imgs, classes = imgs.to(device), classes.to(device)

        # Forward pass
        outputs = model(imgs)
        loss = loss_fn(outputs, classes)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate top-1 and top-5 error rates
        _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct_top1 = top5_preds[:, 0] == classes
        correct_top5 = top5_preds.eq(classes.view(-1, 1)).sum(dim=1) > 0
        top1_error = 1 - correct_top1.sum().item() / classes.size(0)
        top5_error = 1 - correct_top5.sum().item() / classes.size(0)
        
        # Update running statistics
        running_loss += loss.item()
        running_top1_error += top1_error
        running_top5_error += top5_error

        # Update total statistics
        total_loss += loss.item() 
        total_top1_error += top1_error
        total_top5_error += top5_error

        # Log every 1000 batches
        if batch % 1000 == 999:
            avg_running_loss = running_loss / 1000
            avg_top1_error = 100.0 * running_top1_error / 1000
            avg_top5_error = 100.0 * running_top5_error / 1000
                        
            # Log to TensorBoard
            tb_x = epoch_index * len(training_dataloader) + batch + 1
            writer.add_scalar('Loss/train_step', avg_running_loss, tb_x)
            writer.add_scalar('Top-1 error/train_step', avg_top1_error, tb_x)
            writer.add_scalar('Top-5 error/train_step', avg_top5_error, tb_x)

            # print(f'  Batch {tb_x} Loss: {avg_running_loss:.4f} Top-1 error rate: {avg_top1_error:.2f}% Top-5 error rate: {avg_top5_error:.2f}%')
            
            running_loss = 0.0
            running_top1_error = 0.0
            running_top5_error = 0.0
        
        batch += 1

        # if batch > 10:
        #     break

    # Calculate epoch-level metrics
    avg_epoch_loss = total_loss / len(training_dataloader)
    avg_top1_error = 100.0 * total_top1_error / len(training_dataloader)
    avg_top5_error = 100.0 * total_top5_error / len(training_dataloader)

    return avg_epoch_loss, avg_top1_error, avg_top5_error

def validate_one_epoch(
    epoch_index,
    model,
    validation_dataloader,
    loss_fn,
    writer
):
    total_loss = 0.0
    total_top1_error = 0.0
    total_top5_error = 0.0
    running_loss = 0.0
    running_top1_error = 0.0
    running_top5_error = 0.0

    # Ensure model is in evaluation mode
    model.eval()

    with torch.no_grad():  # Disable gradient computation

        batch = 0
        for imgs, classes in tqdm(validation_dataloader, desc='Validation'):
            # Move each crop tensor in the inputs tuple to the device
            imgs = [img.to(device) for img in imgs]
            classes = classes.to(device)

            imgs = torch.stack(imgs, dim=0)

            # Forward pass through the model
            outputs = model(imgs)

            # Calculate loss using outputs
            loss = loss_fn(outputs, classes)

            # Calculate top-1 and top-5 error rates
            _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top1 = top5_preds[:, 0] == classes
            correct_top5 = top5_preds.eq(classes.view(-1, 1)).any(dim=1)
            top1_error = 1 - correct_top1.float().mean().item()
            top5_error = 1 - correct_top5.float().mean().item()
            
            # Update running statistics
            running_loss += loss.item()
            running_top1_error += top1_error
            running_top5_error += top5_error

            # Update total statistics
            total_loss += loss.item()
            total_top1_error += top1_error
            total_top5_error += top5_error

            # Log every 100 batches
            if batch % 100 == 99:
                avg_running_loss = running_loss / 100
                avg_top1_error = 100.0 * running_top1_error / 100
                avg_top5_error = 100.0 * running_top5_error / 100
                
                # print(f'  Validation Batch {batch + 1:5d} Loss: {avg_running_loss:.4f} Top-1 error: {avg_top1_error:.2f}% Top-5 error: {avg_top5_error:.2f}%')
                
                # Log to TensorBoard
                tb_x = epoch_index * len(validation_dataloader) + batch + 1
                writer.add_scalar('Loss/val_step', avg_running_loss, tb_x)
                writer.add_scalar('Top-1 error/val_step', avg_top1_error, tb_x)
                writer.add_scalar('Top-5 error/val_step', avg_top5_error, tb_x)
                
                # Reset running statistics
                running_loss = 0.0
                running_top1_error = 0.0
                running_top5_error = 0.0

            batch += 1

            # if batch > 10:
            #     break

    # Calculate epoch-level metrics
    avg_epoch_loss = total_loss / len(validation_dataloader)
    avg_top1_error = 100.0 * total_top1_error / len(validation_dataloader)
    avg_top5_error = 100.0 * total_top5_error / len(validation_dataloader)
    
    return avg_epoch_loss, avg_top1_error, avg_top5_error

if __name__ == "__main__":
    OUTPUT_DIR = 'alexnet_data_out'

    # print the seed value
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))

    # make checkpoint path directory
    CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device = {device}",)

    LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('Tensorboard summary writer created')

    # create model
    NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    print(alexnet)
    print('AlexNet created')

    # create dataset and data loader
    IMAGE_DIM = 227  # pixels
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    BATCH_SIZE = 128
    DATA_ROOT = '/home/sree/data/ILSVRC2012'
    trainset = torchvision.datasets.ImageNet(
        root=DATA_ROOT,
        split='train',
        transform=preprocess
    )
    print('Train Dataset created')

    trainloader = torch.utils.data.DataLoader(
        trainset,
        shuffle=True,
        pin_memory=False,
        num_workers=4,
        drop_last=True,
        batch_size=BATCH_SIZE,
        # prefetch_factor=8,
        persistent_workers=True,
    )
    print('Train Dataloader created')

    testset = torchvision.datasets.ImageNet(
        root=DATA_ROOT,
        split='val',
        transform=preprocess
    )
    print('Valid Dataset created')

    testloader = torch.utils.data.DataLoader(
        testset,
        shuffle=False,
        pin_memory=False,
        num_workers=2,
        drop_last=True,
        batch_size=BATCH_SIZE,
        # prefetch_factor=2,
        persistent_workers=True,
    )
    print('Test Dataloader created')

    # create optimizer
    # the one that WORKS
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    print('Optimizer created')

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    loss = torch.nn.CrossEntropyLoss()
    print("Loss function created")

    print('Starting training...')
    NUM_EPOCHS = 120
    best_vloss = 1000000.0
    training_start_time = timer()
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = timer()
        # Training phase
        avg_train_loss, avg_train_top1_error, avg_train_top5_error = train_one_epoch(
            epoch,
            trainloader,
            alexnet,
            loss,
            optimizer,
            tbwriter
        )

        # Validation pahse
        avg_val_loss, avg_val_top1_error, avg_val_top5_error = validate_one_epoch(
            epoch,
            alexnet,
            testloader,
            loss,
            tbwriter,
        )

        print(f'LOSS train {avg_train_loss:.4f} valid {avg_val_loss:.4f}')
        print(f'Top-1 Error train {avg_train_top1_error:.2f}% val {avg_val_top1_error:.2f}%')
        print(f'Top-5 Error train {avg_train_top5_error:.2f}% val {avg_val_top5_error:.2f}%')

        # Update scheduler
        lr_scheduler.step()

        # Log epoch-level metrics
        tbwriter.add_scalars('Training vs. Validation Loss',
                        { 'Training': avg_train_loss, 'Validation': avg_val_loss },
                        epoch + 1)
        tbwriter.add_scalars('Training vs. Validation Top-1 Error',
                        { 'Training': avg_train_top1_error, 'Validation': avg_val_top1_error },
                        epoch + 1)
        tbwriter.add_scalars('Training vs. Validation Top-5 Error',
                        { 'Training': avg_train_top5_error, 'Validation': avg_val_top5_error },
                        epoch + 1)
        tbwriter.add_scalar('Learning rate', lr_scheduler.get_last_lr()[0], epoch + 1)
        tbwriter.flush()

        # Track best performance and save model
        if avg_val_loss < best_vloss:
            # save checkpoints
            print(f"Saving check point as avg_val_loss({avg_val_loss}) < best_vloss({best_vloss})")
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': alexnet.state_dict(),
                'seed': seed,
            }
            torch.save(state, checkpoint_path)

        epoch_end_time = timer()
        epoch_elpsed_time = epoch_end_time - epoch_start_time
        print(f"Time for one epoch = {epoch_elpsed_time}") # time in seconds
        # break

    training_end_time = timer()
    training_elapsed_time = training_end_time - training_start_time
    print(f"Time for training = {training_elapsed_time}") # time in seconds
