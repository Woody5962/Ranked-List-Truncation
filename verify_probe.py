import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch as t
import torch.nn as nn
import torch.optim as optim
import logging
import configparser
import shutil 

from dataloader import *
from models import *
from utils import losses
from utils.metrics import Metric

from tensorboardX import SummaryWriter


RUNNING_PATH = '/home/LAB/wangd/graduation_project/ranked list truncation'
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, args):
        """trainer for the truncation model
        
        Args:
            args: args for training
        """
        # params for training
        self.seq_len = 300 if args.retrieve_data == 'robust04' else 40
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.save_path = args.save_path
        self.epochs_base = args.epochs_base
        self.epochs_probe = args.epochs_probe
        self.lr = args.lr
        self.cuda = args.cuda
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.ft = args.ft
        # params for search
        self.parameter_record = args.parameter_record
        self.parameter_search = args.parameter_search
        self.regularizer_search = args.regularizer_search
        self.mt_search = args.mt_search
        # params for evaluation
        self.best_test_f1 = -float('inf')
        self.best_test_dcg = -float('inf')
        self.f1_record = []
        self.dcg_record = []
        
        mmoe_input = 3 if args.retrieve_data == 'robust04' else 47
        self.train_loader, self.test_loader, _ = at_dataloader(args.retrieve_data, args.dataset_name, args.batch_size) if args.retrieve_data == 'robust04' else \
            mc_dataloader(args.retrieve_data, args.dataset_name, args.batch_size)
        
        # base模型及其对应优化器的初始化
        self.model_base = ProbeBase(seq_len=self.seq_len, num_tasks=args.num_tasks, input_size=mmoe_input, dropout=self.dropout, num_experts=2)
        self.criterion_base = losses.MtCutLoss(metric=args.criterion, num_tasks=args.num_tasks)
        self.optimizer_base = optim.Adam(self.model_base.parameters(), lr=args.lr, weight_decay=self.weight_decay)
        
        # probe模型的初始化
        self.probe_c1 = TowerClass(d_model=256)
        self.probe_r1 = TowerRerank(d_model=256)
        self.probe_ce1 = TowerClass(d_model=256)
        self.probe_ce2 = TowerClass(d_model=256)
        self.probe_re1 = TowerRerank(d_model=256)
        self.probe_re2 = TowerRerank(d_model=256)
        
        # probe模型对应优化器的初始化
        self.optimizer_c1 = optim.Adam(self.probe_c1.parameters(), lr=args.lr, weight_decay=self.weight_decay)
        self.optimizer_r1 = optim.Adam(self.probe_r1.parameters(), lr=args.lr, weight_decay=self.weight_decay)
        self.optimizer_ce1 = optim.Adam(self.probe_ce1.parameters(), lr=args.lr, weight_decay=self.weight_decay)
        self.optimizer_ce2 = optim.Adam(self.probe_ce2.parameters(), lr=args.lr, weight_decay=self.weight_decay)
        self.optimizer_re1 = optim.Adam(self.probe_re1.parameters(), lr=args.lr, weight_decay=self.weight_decay)
        self.optimizer_re2 = optim.Adam(self.probe_re2.parameters(), lr=args.lr, weight_decay=self.weight_decay)
        
        # 两类probe对应的criterion
        self.criterion_probe_c = nn.BCELoss()
        self.criterion_probe_r = losses.RerankLoss()
        
        if self.cuda: 
            self.model_base = self.model.cuda()
            self.criterion = self.criterion.cuda()
        
        if args.ft and os.path.exists(self.model_path): self.load_model()
        self.writer = SummaryWriter(log_dir=args.Tensorboard_dir, comment='probe')
            
    def train_base(self, epoch):
        """train stage for every epoch
        """
        start_time = time.time()
        epoch_loss, epoch_f1, epoch_dcg = 0, 0, 0
        step = 0
        logging.info('-' * 100)
        
        # train the Base model which will produce the information for probing
        for X_train, y_train in tqdm(self.train_loader, desc='Training for epoch_{}'.format(epoch)):
            self.model_base.train()
            self.optimizer_base.zero_grad()
            if self.cuda: X_train, y_train = X_train.cuda(), y_train.cuda()

            output = self.model_base(X_train)
            loss = self.criterion_base(output[-1], y_train)

            loss.backward()
            self.optimizer_base.step()
            
            predictions = output[-1][-1].detach().cpu().squeeze().numpy()
            k_s = np.argmax(predictions, axis=1) + 1
            
            y_train = y_train.data.cpu().numpy()
            f1 = Metric.f1(y_train, k_s)
            dcg = Metric.dcg(y_train, k_s)
            self.writer.add_scalar('train_base/loss_step', loss.item(), step + self.batch_size * epoch)

            epoch_loss += loss.item()
            epoch_f1 += f1
            epoch_dcg += dcg
            step += 1

        train_loss, train_f1, train_dcg = epoch_loss / step, epoch_f1 / step, epoch_dcg / step
        self.writer.add_scalar('train_base/loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train_base/F1_epoch', train_f1, epoch)
        self.writer.add_scalar('train_base/DCG_epoch', train_dcg, epoch)
        logging.info('\nEpoch: {} | Epoch Time: {:.2f} s'.format(epoch, time.time() - start_time))
        logging.info('\tTrain: loss = {} | f1 = {:.6f} | dcg = {:.6f}\n'.format(train_loss, train_f1, train_dcg))

    def test_base(self, epoch):
        """test stage for every epoch
        """
        epoch_loss, epoch_f1, epoch_dcg = 0, 0, 0
        step = 0
        for X_test, y_test in tqdm(self.test_loader, desc='Test after epoch_{}'.format(epoch)):
            self.model_base.eval()
            with t.no_grad():
                if self.cuda: X_test, y_test = X_test.cuda(), y_test.cuda()
                output = self.model_base(X_test)
                loss = self.criterion_base(output[-1], y_test)
                
                predictions = output[-1][-1].detach().cpu().squeeze().numpy()
                k_s = np.argmax(predictions, axis=1) + 1

                y_test = y_test.data.cpu().numpy()
                f1 = Metric.f1(y_test, k_s)
                dcg = Metric.dcg(y_test, k_s)
                
                epoch_loss += loss.item()
                epoch_f1 += f1
                epoch_dcg += dcg
                step += 1
        
        test_loss, test_f1, test_dcg = epoch_loss / step, epoch_f1 / step, epoch_dcg / step
        self.writer.add_scalar('test_base/loss_epoch', test_loss, epoch)
        self.writer.add_scalar('test_base/F1_epoch', test_f1, epoch)
        self.writer.add_scalar('test_base/DCG_epoch', test_dcg, epoch)
        self.f1_record.append(test_f1)
        self.dcg_record.append(test_dcg)
        logging.info('\tTest: loss = {} | f1 = {:.6f} | dcg = {:.6f}\n'.format(test_loss, test_f1, test_dcg))
        
        if test_f1 > self.best_test_f1:
            self.best_test_f1 = test_f1
            self.save_model()
        if test_dcg > self.best_test_dcg: self.best_test_dcg = test_dcg
    
    
    def train_probe(self, epoch):
        """train the probing network

        Args:
            epoch (int): epochs for training the probing network 
        """
        # probing for the pre_encoding layer
        step = 0
        logging.info('-' * 100)
        for X_train, y_train in tqdm(self.train_loader, desc='Test after epoch_{}'.format(epoch)):
            if self.cuda: X_train, y_train = X_train.cuda(), y_train.cuda()
            with t.no_grad():
                output = self.model_base(X_train)
                experts_in, experts_o = output[:2]
            # probing networks
            probe_c1 = self.probe_c1(experts_in).squeeze()
            probe_r1 = self.probe_r1(experts_in).squeeze()
            probe_ce1 = self.probe_ce1(experts_o[0]).squeeze()
            probe_ce2 = self.probe_ce2(experts_o[1]).squeeze()
            probe_re1 = self.probe_re1(experts_o[0]).squeeze()
            probe_re2 = self.probe_re2(experts_o[1]).squeeze()
            
            # losses
            loss_c1 = self.criterion_probe_c(probe_c1, y_train)
            loss_r1 = self.criterion_probe_r(probe_r1, y_train)
            loss_ce1 = self.criterion_probe_c(probe_ce1, y_train)
            loss_ce2 = self.criterion_probe_c(probe_ce2, y_train)
            loss_re1 = self.criterion_probe_r(probe_re1, y_train)
            loss_re2 = self.criterion_probe_r(probe_re2, y_train)
            
            # optimizer
            loss_c1.backward()
            self.optimizer_c1.step()
            loss_r1.backward()
            self.optimizer_r1.step()
            loss_ce1.backward()
            self.optimizer_ce1.step()
            loss_ce2.backward()
            self.optimizer_ce2.step()
            loss_re1.backward()
            self.optimizer_re1.step()
            loss_re2.backward()
            self.optimizer_re2.step()
            
            # metric for probing networks
            metric_c1 = Metric.taskc_metric(y_train, probe_c1.detach().numpy())
            metric_r1 = Metric.taskr_metric(y_train, probe_r1.detach().numpy())
            metric_ce1 = Metric.taskc_metric(y_train, probe_ce1.detach().numpy())
            metric_re1 = Metric.taskr_metric(y_train, probe_re1.detach().numpy())
            metric_ce2 = Metric.taskc_metric(y_train, probe_ce2.detach().numpy())
            metric_re2 = Metric.taskr_metric(y_train, probe_re2.detach().numpy())
    
            # probing record
            self.writer.add_scalar('probe/pre_encoding_classification', metric_c1, step + self.batch_size * epoch)
            self.writer.add_scalar('probe/pre_encoding_rerank', metric_r1, step + self.batch_size * epoch)
            self.writer.add_scalar('probe/expert0_classification', metric_ce1, step + self.batch_size * epoch)
            self.writer.add_scalar('probe/expert0_rerank', metric_re1, step + self.batch_size * epoch)
            self.writer.add_scalar('probe/expert1_classification', metric_ce2, step + self.batch_size * epoch)
            self.writer.add_scalar('probe/expert1_rerank', metric_re2, step + self.batch_size * epoch)

    def save_model(self):
        """save the best model
        """
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        t.save(self.model_base.state_dict(), self.save_path + '{}.pkl'.format(self.model_name))
        logging.info('The best model has beed updated and saved in {}\n'.format(self.save_path))

    def load_model(self):
        """load the saved model
        """
        self.model_base.load_state_dict(t.load(self.model_path))
        logging.info('The best model has beed loaded from {}\n'.format(self.model_path))

    def run(self):
        """run the model
        """
        if self.ft: self.load_model()
        else:
            logging.info('\nTrain the Base model: \n')
            for epoch in range(self.epochs_base):
                self.train_base(epoch)
                self.test_base(epoch)
            best5_f1 = sum(sorted(self.f1_record, reverse=True)[:5]) / 5
            best5_dcg = sum(sorted(self.dcg_record, reverse=True)[:5]) / 5
            logging.info('the best metric of this model: f1: {} | dcg: {}'.format(self.best_test_f1, self.best_test_dcg))
            logging.info('the best-5 metric of this model: f1: {} | dcg: {}'.format(best5_f1, best5_dcg))
        
        for epoch in range(self.epochs_probe):
            self.train_probe(epoch)


def main():
    """训练过程的主函数，用于接收训练参数等
    """
    parser = argparse.ArgumentParser(description="Truncation Model Trainer Args")
    parser.add_argument('--retrieve-data', type=str, default='robust04')
    parser.add_argument('--dataset-name', type=str, default='drmm_tks')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--model-name', type=str, default='probe_base')
    parser.add_argument('--criterion', type=str, default='f1')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--ft', type=int, default=1)
    parser.add_argument('--save-path', type=str, default='{}/best_model/'.format(RUNNING_PATH))
    parser.add_argument('--epochs-base', type=int, default=20)
    parser.add_argument('--epochs-probe', type=int, default=180)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--parameter-record', type=str, default='{}/parameters.log'.format(RUNNING_PATH))
    parser.add_argument('--parameter-search', type=int, default=0)
    parser.add_argument('--regularizer-search', type=int, default=0)
    parser.add_argument('--mt-search', type=int, default=0)
    parser.add_argument('--search-times', type=int, default=80)
    parser.add_argument('--num-tasks', type=float, default=3)  # 2.1:classification + truncation | 2.2: rerank + truncation
    parser.add_argument('--rerank-weight', type=float, default=0.5)
    parser.add_argument('--class-weight', type=float, default=0.8)

    args = parser.parse_args()
    args.cuda = t.cuda.is_available()
    args.Tensorboard_dir = '{}/Tensorboard_summary/Probe'.format(RUNNING_PATH)
    args.model_path = args.save_path + '{}.pkl'.format(args.model_name)
    
    if os.path.exists(args.Tensorboard_dir): shutil.rmtree(args.Tensorboard_dir)
    os.makedirs(args.Tensorboard_dir)
    
    config = configparser.ConfigParser()
    config.read('{}/hyper_parameter_{}.conf'.format(RUNNING_PATH, args.dataset_name))
    args.lr = config.getfloat('{}_conf'.format(args.model_name), 'lr')
    if args.retrieve_data == 'robust04': args.batch_size = config.getint('{}_conf'.format(args.model_name), 'batch_size')
    args.dropout = config.getfloat('{}_conf'.format(args.model_name), 'dropout')
    args.weight_decay = config.getfloat('{}_conf'.format(args.model_name), 'weight_decay')
    args.rerank_weight = config.getfloat('{}_conf'.format(args.model_name), 'rerank_weight')
    args.class_weight = config.getfloat('{}_conf'.format(args.model_name), 'class_weight')

    logging.info('{}'.format(vars(args)))
    trainer = Trainer(args)
    trainer.run()
    trainer.writer.close()

if __name__ == '__main__':
    main()
