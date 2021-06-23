import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch as t
import torch.optim as optim

from dataloader import bc_dataloader, cp_dataloader, at_dataloader
from models import BiCut, Choopy, AttnCut, MTCut
from utils import losses
from utils.metrics import Metric

from tensorboardX import SummaryWriter


RUNNING_PATH = '/home/LAB/wangd/graduation_project/ranked list truncation'


class Trainer:
    def __init__(self, args):
        """trainer for the truncation model

        Args:
            args: args for training
        """
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.lr = args.lr
        self.cuda = args.cuda
        self.weight_decay = args.weight_decay

        if self.model_name == 'bicut':
            self.train_loader, self.test_loader, _ = at_dataloader(args.dataset_name, args.batch_size)
            # self.train_loader, self.test_loader = bc_dataloader(args.dataset_name, args.batch_size)
            self.model = BiCut(input_size=5)
            self.criterion = losses.BiCutLoss(args.criterion)
        elif self.model_name == 'choopy':
            self.train_loader, self.test_loader, _ = cp_dataloader(args.dataset_name, args.batch_size)
            self.model = Choopy()
            self.criterion = losses.ChoopyLoss(metric=args.criterion)
        elif self.model_name == 'attncut':
            self.train_loader, self.test_loader, _ = at_dataloader(args.dataset_name, args.batch_size)
            self.model = AttnCut()
            self.criterion = losses.AttnCutLoss(metric=args.criterion)
        elif self.model_name == 'mtcut':
            self.train_loader, self.test_loader, _ = at_dataloader(args.dataset_name, args.batch_size)
            self.model = MTCut()
            self.criterion = losses.MTCutLoss(metric=args.criterion)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=self.weight_decay)
        self.best_test_metric = float('inf')
        
        if self.cuda: 
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        
        if args.ft and os.path.exists(self.model_path): self.load_model()
        self.writer = SummaryWriter(log_dir=args.Tensorboard_dir)
            
    def train_epoch(self, epoch):
        """train stage for every epoch
        """
        start_time = time.time()
        epoch_loss, epoch_f1, epoch_dcg = 0, 0, 0
        step, num_itr = 0, len(self.train_loader)
        print('-' * 100)
        for X_train, y_train in tqdm(self.train_loader, desc='Training for epoch_{}'.format(epoch)):
            self.model.train()
            self.optimizer.zero_grad()
            if self.cuda: X_train, y_train = X_train.cuda(), y_train.cuda()

            output = self.model(X_train)
            loss = self.criterion(output, y_train)

            loss.backward()
            self.optimizer.step()
            
            if self.model_name == 'bicut':
                predictions = np.argmax(output.detach().cpu().numpy(), axis=2)
                k_s = []
                for results in predictions:
                    if np.sum(results) == 300: k_s.append(300)
                    else: k_s.append(np.argmin(results))
            elif self.model_name == 'mtcut':
                predictions = output[1].detach().cpu().squeeze().numpy()
                k_s = np.argmax(predictions, axis=1)
            else: 
                predictions = output.detach().cpu().squeeze().numpy()
                k_s = np.argmax(predictions, axis=1)
            y_train = y_train.data.cpu().numpy()
            f1 = Metric.f1(y_train, k_s)
            dcg = Metric.dcg(y_train, k_s)
            self.writer.add_scalar('train/loss_step', loss.item(), step + num_itr * epoch)

            epoch_loss += loss.item()
            epoch_f1 += f1
            epoch_dcg += dcg
            step += 1

        train_loss, train_f1, train_dcg = epoch_loss / step, epoch_f1 / step, epoch_dcg / step
        self.writer.add_scalar('train/loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/F1_epoch', train_f1, epoch)
        self.writer.add_scalar('train/DCG_epoch', train_dcg, epoch)
        print('\nEpoch: {} | Epoch Time: {:.2f} s'.format(epoch, time.time() - start_time))
        print('\tTrain: loss = {:.2f}, f1 = {:.4f}, dcg = {:.4f}\n'.format(train_loss, train_f1, train_dcg))

    def test(self, epoch):
        """test stage for every epoch
        """
        epoch_loss, epoch_f1, epoch_dcg = 0, 0, 0
        step = 0
        for X_test, y_test in tqdm(self.test_loader, desc='Test after epoch_{}'.format(epoch)):
            self.model.eval()
            with t.no_grad():
                if self.cuda: X_test, y_test = X_test.cuda(), y_test.cuda()
                output = self.model(X_test)
                loss = self.criterion(output, y_test)
                
                if self.model_name == 'bicut':
                    predictions = np.argmax(output.detach().cpu().numpy(), axis=2)
                    k_s = []
                    for results in predictions:
                        if np.sum(results) == 300: k_s.append(300)
                        else: k_s.append(np.argmin(results))
                elif self.model_name == 'mtcut':
                    predictions = output[1].detach().cpu().squeeze().numpy()
                    k_s = np.argmax(predictions, axis=1)
                else: 
                    predictions = output.detach().cpu().squeeze().numpy()
                    k_s = np.argmax(predictions, axis=1)
                y_test = y_test.data.cpu().numpy()
                f1 = Metric.f1(y_test, k_s)
                dcg = Metric.dcg(y_test, k_s)
                
                epoch_loss += loss.item()
                epoch_f1 += f1
                epoch_dcg += dcg
                step += 1
        
        test_loss, test_f1, test_dcg = epoch_loss / step, epoch_f1 / step, epoch_dcg / step
        self.writer.add_scalar('test/loss_epoch', test_loss, epoch)
        self.writer.add_scalar('test/F1_epoch', test_f1, epoch)
        self.writer.add_scalar('test/DCG_epoch', test_dcg, epoch)
        print('\tTest: loss = {:.2f}, f1 = {:.4f}, dcg = {:.4f}'.format(test_loss, test_f1, test_dcg))
        
        if test_f1 > self.best_test_metric:
            self.best_test_metric = test_f1
            self.save_model()

    def save_model(self):
        """save the best model
        """
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        t.save(self.model.state_dict(), self.save_path + '{}.pkl'.format(self.model_name))
        print('The best model has beed updated and saved in {}\n'.format(self.save_path))

    def load_model(self):
        """load the saved model
        """
        self.model.load_state_dict(t.load(self.model_path))
        print('The best model has beed loaded from {}\n'.format(self.model_path))

    def run(self):
        """run the model
        """
        print('\nTrain the {} model: \n'.format(self.model_name))
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            self.test(epoch)

def main():
    parser = argparse.ArgumentParser(description="Truncation Model Trainer Args")
    parser.add_argument('--dataset-name', type=str, default='drmm')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model-name', type=str, default='choopy')
    parser.add_argument('--criterion', type=str, default='f1')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--ft', type=bool, default=False)
    parser.add_argument('--save-path', type=str, default='{}/best_model/'.format(RUNNING_PATH))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.015)

    args = parser.parse_args()
    args.cuda = t.cuda.is_available()
    print('Using GPU: {}'.format(args.cuda))
    args.Tensorboard_dir = '{}/Tensorboard_summary/'.format(RUNNING_PATH)
    args.model_path = args.save_path + '{}.pkl'.format(args.model_name)
    
    if not os.path.exists(args.Tensorboard_dir):
        os.mkdir(args.Tensorboard_dir)
    if args.model_name == 'attncut':
        args.lr, args.batch_size = 3e-5, 20
    elif args.model_name == 'choopy':
        args.lr, args.batch_size = 1e-3, 63
    elif args.model_name == 'bicut':
        args.lr, args.batch_size = 1e-4, 32

    trainer = Trainer(args)
    trainer.run()
    trainer.writer.close()
    

if __name__ == '__main__':
    main()
