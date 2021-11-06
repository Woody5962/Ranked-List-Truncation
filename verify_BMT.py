import time
import argparse
import numpy as np
from tqdm import tqdm
import torch as t
import torch.optim as optim
import logging
import configparser

from sklearn import metrics

from dataloader import *
from models import *
from utils import Metric
from utils import losses


RUNNING_PATH = '/home/LAB/wangd/graduation_project/ranked list truncation'
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, args):
        """trainer for the truncation model

        Args:
            args: args for training
        """
        # params for training
        self.verify_type = args.verify_type
        self.seq_len = 300 if args.retrieve_data == 'robust04' else 40
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.epochs = args.epochs
        self.lr = args.lr
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        self.ft = args.ft
        self.metric = []
        self.metric_name = 'auc' if self.verify_type == 'c' else 'DCG'

        input_size = 3 if args.retrieve_data == 'robust04' else 25
        if self.model_name == 'choopy':
            self.train_loader, self.test_loader, _ = cp_dataloader(args.retrieve_data, args.dataset_name, args.batch_size)
            self.cut_model = Choopy(seq_len=self.seq_len, dropout=self.dropout)
            if args.verify_type == 'c': self.model = TaskC(d_model=128) if self.ft == 1 else TaskC(d_model=input_size)
            else: self.model = TaskR(d_model=128) if self.ft == 1 else TaskR(d_model=input_size)
        elif self.model_name == 'attncut':
            attncut_input = 3 if args.retrieve_data == 'robust04' else 25
            self.train_loader, self.test_loader, _ = at_dataloader(args.retrieve_data, args.dataset_name, args.batch_size)
            self.cut_model = AttnCut(input_size=attncut_input, dropout=self.dropout)
            if args.verify_type == 'c': self.model = TaskC(d_model=256) if self.ft == 1 else TaskC(d_model=input_size)
            else: self.model = TaskR(d_model=256) if self.ft == 1 else TaskR(d_model=input_size)
        
        if self.ft: self.load_model()
        self.criterion = t.nn.BCELoss() if args.verify_type == 'c' else losses.RerankLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=self.weight_decay)
            
    def train_epoch(self, epoch):
        """train stage for every epoch
        """
        start_time = time.time()
        epoch_loss, epoch_metric = 0, 0
        step = 0
        logging.info('-' * 100)
        for X_train, y_train in tqdm(self.train_loader, desc='Training for epoch_{}'.format(epoch)):
            if self.ft: 
                with t.no_grad():
                    if self.model_name == 'attncut': 
                        X_train = self.cut_model.encoding_layer(X_train)[0]
                        X_train = self.cut_model.attention_layer(X_train) 
                    elif self.model_name == 'choopy':
                        pe = self.cut_model.position_encoding.expand(X_train.shape[0], self.seq_len, 127)
                        X_train = t.cat((X_train, pe), dim=2)
                        X_train = self.attention_layer(X_train)
            self.model.train()
            self.optimizer.zero_grad()

            output = self.model(X_train)
            loss = self.criterion(output.squeeze(), y_train)

            loss.backward()
            self.optimizer.step()
            
            y_train = y_train.detach().numpy()
            output = output.detach().numpy().squeeze()
            
            if self.verify_type == 'c': step_metric = Metric.taskc_metric(y_train, output)
            else: step_metric = Metric.taskr_metric(y_train, output)

            epoch_loss += loss.item()
            epoch_metric += step_metric
            step += 1
            if len(self.metric) < 100: self.metric.append(step_metric)

        train_loss, train_metric = epoch_loss / step, epoch_metric / step
        logging.info('\nEpoch: {} | Epoch Time: {:.2f} s'.format(epoch, time.time() - start_time))
        logging.info('\tTrain: loss = {} | {} = {:.6f}\n'.format(train_loss, self.metric_name, train_metric))

    def test(self, epoch):
        """test stage for every epoch
        """
        epoch_loss, epoch_metric = 0, 0
        step = 0
        for X_test, y_test in tqdm(self.test_loader, desc='Test after epoch_{}'.format(epoch)):
            self.model.eval()
            with t.no_grad():
                if self.ft: 
                    if self.model_name == 'attncut': 
                        X_test = self.cut_model.encoding_layer(X_test)[0]
                        X_test = self.cut_model.attention_layer(X_test)
                    elif self.model_name == 'choopy':
                        pe = self.cut_model.position_encoding.expand(X_test.shape[0], self.seq_len, 127)
                        X_test = t.cat((X_test, pe), dim=2)
                        X_test = self.attention_layer(X_test)
                output = self.model(X_test)
                loss = self.criterion(output.squeeze(), y_test)
                
                y_test = y_test.detach().numpy()
                output = output.detach().numpy().squeeze()
                
                if self.verify_type == 'c': step_metric = Metric.taskc_metric(y_test, output)
                else: step_metric = Metric.taskr_metric(y_test, output)

                epoch_loss += loss.item()
                epoch_metric += step_metric
                step += 1
        
        test_loss, test_metric = epoch_loss / step, epoch_metric / step
        logging.info('\tTest: loss = {} | {} = {:.6f}\n'.format(test_loss, self.metric_name, test_metric))
        

    def load_model(self):
        """load the saved model
        """
        self.cut_model.load_state_dict(t.load(self.model_path))
        logging.info('The best model has beed loaded from {}\n'.format(self.model_path))

    def run(self):
        """run the model
        """
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            self.test(epoch)
        print(self.metric)
        


def main():
    """训练过程的主函数，用于接收训练参数等
    """
    parser = argparse.ArgumentParser(description="Verification Trainer Args")
    parser.add_argument('--retrieve-data', type=str, default='robust04')
    parser.add_argument('--dataset-name', type=str, default='drmm_tks')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--model-name', type=str, default='attncut')
    parser.add_argument('--verify-type', type=str, default='r')  # c: classification, r: rerank
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--save-path', type=str, default='{}/best_model/'.format(RUNNING_PATH))
    parser.add_argument('--ft', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=0.0015)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()
    args.model_path = args.save_path + '{}.pkl'.format(args.model_name)
    

    logging.info('{}'.format(vars(args)))
    trainer = Trainer(args)
    trainer.run()

if __name__ == '__main__':
    main()
