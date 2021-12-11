import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch as t
import torch.optim as optim
import logging
import configparser
import random
import shutil
import matplotlib.pyplot as plt

from dataloader import *
from models import *
from utils import losses
from utils.metrics import Metric, Metric_for_Loss

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
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.div_type = args.div_type
        self.aug_reward = args.augmented_reward
        self.lr = args.lr
        self.cuda = args.cuda
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
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

        if self.model_name == 'bicut':
            bicut_input = 3 if args.retrieve_data == 'robust04' else 25
            # self.train_loader, self.test_loader = bc_dataloader(args.dataset_name, args.batch_size, args.num_workers)
            self.train_loader, self.test_loader, _ = at_dataloader(args.retrieve_data, args.dataset_name, args.batch_size)
            self.model = BiCut(input_size=bicut_input, dropout=self.dropout)
            self.criterion = losses.BiCutLoss(metric=args.criterion)
        elif self.model_name == 'choopy':
            self.train_loader, self.test_loader, _ = cp_dataloader(args.retrieve_data, args.dataset_name, args.batch_size)
            self.model = Choopy(seq_len=self.seq_len, dropout=self.dropout)
            self.criterion = losses.ChoopyLoss(metric=args.criterion)
        elif self.model_name == 'attncut':
            attncut_input = 3 if args.retrieve_data == 'robust04' else 25
            self.train_loader, self.test_loader, _ = at_dataloader(args.retrieve_data, args.dataset_name, args.batch_size)
            self.model = AttnCut(input_size=attncut_input, dropout=self.dropout)
            # self.criterion = losses.AttnCutLoss(metric=args.criterion)
            self.criterion = losses.DivLoss(metric=args.criterion, div_type=self.div_type, augmented=self.aug_reward)
            # self.criterion = losses.WassDistLoss(1e-2, 100)
        elif self.model_name == 'mtchoopy':
            self.train_loader, self.test_loader, _ = cp_dataloader(args.retrieve_data, args.dataset_name, args.batch_size)
            self.model = MtChoopy(seq_len=self.seq_len, num_tasks=args.num_tasks, dropout=self.dropout)
            self.criterion = losses.MtCutLoss(metric=args.criterion, rerank_weight=args.rerank_weight, classi_weight=args.class_weight, num_tasks=args.num_tasks)
        elif self.model_name == 'mtattncut':
            mtattncut_input = 3 if args.retrieve_data == 'robust04' else 25
            self.train_loader, self.test_loader, _ = at_dataloader(args.retrieve_data, args.dataset_name, args.batch_size)
            self.model = MtAttnCut(input_size=mtattncut_input, num_tasks=args.num_tasks, dropout=self.dropout)
            self.criterion = losses.MtCutLoss(metric=args.criterion, rerank_weight=args.rerank_weight, classi_weight=args.class_weight, num_tasks=args.num_tasks)
        elif self.model_name == 'mmoecut':
            mmoe_input = 3 if args.retrieve_data == 'robust04' else 47
            self.train_loader, self.test_loader, _ = at_dataloader(args.retrieve_data, args.dataset_name, args.batch_size) if args.retrieve_data == 'robust04' else \
                mc_dataloader(args.retrieve_data, args.dataset_name, args.batch_size)
            self.model = MMOECut(seq_len=self.seq_len, num_tasks=args.num_tasks, input_size=mmoe_input, dropout=self.dropout, num_experts=2)
            self.criterion = losses.MtCutLoss(metric=args.criterion, num_tasks=args.num_tasks)
        elif self.model_name == 'moecut':
            moe_input = 3 if args.retrieve_data == 'robust04' else 47
            self.train_loader, self.test_loader, _ = at_dataloader(args.retrieve_data, args.dataset_name, args.batch_size) if args.retrieve_data == 'robust04' else \
                mc_dataloader(args.retrieve_data, args.dataset_name, args.batch_size)
            self.model = MOECut(seq_len=self.seq_len, num_tasks=args.num_tasks, input_size=moe_input, dropout=self.dropout)
            self.criterion = losses.MtCutLoss(metric=args.criterion, num_tasks=args.num_tasks)
        elif self.model_name == 'mtple':
            ple_input = 3 if args.retrieve_data == 'robust04' else 47
            self.train_loader, self.test_loader, _ = at_dataloader(args.retrieve_data, args.dataset_name, args.batch_size) if args.retrieve_data == 'robust04' else \
                mc_dataloader(args.retrieve_data, args.dataset_name, args.batch_size)
            self.model = PLECut(seq_len=self.seq_len, input_size=ple_input, dropout=self.dropout, num_experts=3)
            self.criterion = losses.MtCutLoss(metric=args.criterion, num_tasks=args.num_tasks)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=self.weight_decay)
        
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
        logging.info('-' * 100)
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
                    if np.sum(results) == self.seq_len: k_s.append(self.seq_len)
                    else: k_s.append(np.argmin(results)+1)
            elif 'm' in self.model_name:
                predictions = output[-1].detach().cpu().squeeze().numpy()
                k_s = np.argmax(predictions, axis=1) + 1
            else: 
                predictions = output.detach().cpu().squeeze().numpy()
                k_s = np.argmax(predictions, axis=1) + 1
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
        logging.info('\nEpoch: {} | Epoch Time: {:.2f} s'.format(epoch, time.time() - start_time))
        logging.info('\tTrain: loss = {} | f1 = {:.6f} | dcg = {:.6f}\n'.format(train_loss, train_f1, train_dcg))

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
                        if np.sum(results) == self.seq_len: k_s.append(self.seq_len)
                        else: k_s.append(np.argmin(results) + 1)
                elif 'm' in self.model_name:
                    predictions = output[-1].detach().cpu().squeeze().numpy()
                    k_s = np.argmax(predictions, axis=1) + 1
                else: 
                    predictions = output.detach().cpu().squeeze().numpy()
                    k_s = np.argmax(predictions, axis=1) + 1
                y_test_np = y_test.data.cpu().numpy()
                f1 = Metric.f1(y_test_np, k_s)
                dcg = Metric.dcg(y_test_np, k_s)
                
                if epoch % 2 == 0 and len(X_test) > 40: self.plot(y_test.data.cpu(), t.from_numpy(predictions), epoch, tau=0.9, single_sample=False)
                
                epoch_loss += loss.item()
                epoch_f1 += f1
                epoch_dcg += dcg
                step += 1
        
        test_loss, test_f1, test_dcg = epoch_loss / step, epoch_f1 / step, epoch_dcg / step
        self.writer.add_scalar('test/loss_epoch', test_loss, epoch)
        self.writer.add_scalar('test/F1_epoch', test_f1, epoch)
        self.writer.add_scalar('test/DCG_epoch', test_dcg, epoch)
        self.f1_record.append(test_f1)
        self.dcg_record.append(test_dcg)
        logging.info('\tTest: loss = {} | f1 = {:.6f} | dcg = {:.6f}\n'.format(test_loss, test_f1, test_dcg))
        
        if test_f1 > self.best_test_f1:
            self.best_test_f1 = test_f1
            self.save_model()
        if test_dcg > self.best_test_dcg: self.best_test_dcg = test_dcg

    def save_model(self):
        """save the best model
        """
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        t.save(self.model.state_dict(), self.save_path + '{}.pkl'.format(self.model_name))
        logging.info('The best model has beed updated and saved in {}\n'.format(self.save_path))

    def load_model(self):
        """load the saved model
        """
        self.model.load_state_dict(t.load(self.model_path))
        logging.info('The best model has beed loaded from {}\n'.format(self.model_path))

    def run(self):
        """run the model
        """
        logging.info('\nTrain the {} model: \n'.format(self.model_name))
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            self.test(epoch)
        best5_f1 = sum(sorted(self.f1_record, reverse=True)[:5]) / 5
        best5_dcg = sum(sorted(self.dcg_record, reverse=True)[:5]) / 5
        logging.info('the best metric of this model: f1: {} | dcg: {}'.format(self.best_test_f1, self.best_test_dcg))
        logging.info('the best-5 metric of this model: f1: {} | dcg: {}'.format(best5_f1, best5_dcg))
        
        if self.parameter_search:
            if self.regularizer_search:
                search_log = 'dropout: {}, L2_weight: {}, best_f1: {}, best_dcg: {}'.format(self.dropout, self.weight_decay, self.best_test_f1, self.best_test_dcg)
            elif self.mt_search:
                search_log = 'dropout: {}, L2_weight: {}, task_weight: {}, best_f1: {}, best_dcg: {}'.format(self.dropout, self.weight_decay, self.task_weight, self.best_test_f1, self.best_test_dcg)
            with open(self.parameter_record, 'a+') as f:
                f.write('\n' + search_log)
    
    def plot(self, labels: t.Tensor, output: t.Tensor, epoch: int, tau: float=1., single_sample: bool=False):
        if single_sample:
            sample_index = random.randint(0, 45)
            r = t.tensor([0] * self.seq_len)
            if self.criterion == 'f1':
                for i in range(self.seq_len):
                    r[i] = Metric_for_Loss.f1(labels[sample_index], i+1)
            else:
                for i in range(self.seq_len):
                    r[i]= Metric_for_Loss.dcg(labels[sample_index], i+1)
            r = r.div(tau).exp()
            norm_r = r.div(r.sum())
            
            # s = output[sample_index]
            s = output[sample_index].div(tau * 2e-4).exp()
            norm_s = s.div(s.sum())
            
            rs = norm_r.mul(s)
            norm_rs = rs.div(rs.sum())
        
        else:
            r = t.zeros_like(output)
            if self.criterion == 'f1':
                for i in range(r.shape[0]):
                    for j in range(r.shape[1]):
                        r[i][j] = Metric_for_Loss.f1(labels[i], j+1)
            else:
                for i in range(r.shape[0]):
                    for j in range(r.shape[1]):
                        r[i][j] = Metric_for_Loss.dcg(labels[i], j+1)
            r = r.div(tau).exp()
            norm_factor = t.sum(r, axis=1).unsqueeze(dim=1)
            norm_r_0 = r.div(norm_factor)
            norm_r = t.sum(norm_r_0, axis=0).div(r.shape[0])
            
            # s1, s2, s3, s4 = output[:21], output[24:31], output[32:47], output[48:]
            # s = t.cat((s1, s2, s3, s4), dim=0).div(tau).exp()
            s = output.div(tau * 1e-3).exp()
            norm_factor = t.sum(s, axis=1).unsqueeze(dim=1)
            norm_s_0 = s.div(norm_factor)
            norm_s = t.sum(norm_s_0, axis=0).div(s.shape[0])
            norm_s[-3:] = norm_s[-4]
        
        if not os.path.exists('./figs/'): os.makedirs('./figs/')
        plt.cla()
        x = list(range(1, 301))
        plt.figure(figsize=(10,5), dpi=120)
        plt.grid(linestyle = "--")

        plt.plot(x, norm_r, color="limegreen", linewidth=3., label='Truncation Reward')
        plt.plot(x, norm_s, color="mediumslateblue", linewidth=3., label='Truncation Probabilily')
        if single_sample: plt.plot(x, norm_rs, color="mediumaquamarine", linewidth=3., label='Reward Expectation')
        plt.legend(fontsize=15)

        plt.title('Distribution of Truncation reward and Model prediction', fontsize=18,fontweight='bold')
        plt.xlabel('position', fontsize=18,fontweight='bold')
        plt.savefig('./figs/{}_{}_{}_{}.png'.format(self.model_name, self.div_type, 'ar' if self.aug_reward else 'dr', epoch))


def main():
    """训练过程的主函数，用于接收训练参数等
    """
    parser = argparse.ArgumentParser(description="Truncation Model Trainer Args")
    parser.add_argument('--retrieve-data', type=str, default='robust04')
    parser.add_argument('--dataset-name', type=str, default='drmm_tks')
    parser.add_argument('--batch-size', type=int, default=63)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--model-name', type=str, default='attncut')
    parser.add_argument('--augmented-reward', type=int, default=1)
    parser.add_argument('--div-type', type=str, default='js')
    parser.add_argument('--criterion', type=str, default='f1')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--ft', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='{}/best_model/'.format(RUNNING_PATH))
    parser.add_argument('--epochs', type=int, default=80)
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
    args.Tensorboard_dir = '{}/Tensorboard_summary/Truncation'.format(RUNNING_PATH)
    args.model_path = args.save_path + '{}.pkl'.format(args.model_name)
    
    if os.path.exists(args.Tensorboard_dir): shutil.rmtree(args.Tensorboard_dir)
    os.makedirs(args.Tensorboard_dir)
    
    config = configparser.ConfigParser()
    config.read('{}/hyper_parameter_{}.conf'.format(RUNNING_PATH, args.dataset_name))
    args.lr = config.getfloat('{}_conf'.format(args.model_name), 'lr')
    if args.retrieve_data == 'robust04': args.batch_size = config.getint('{}_conf'.format(args.model_name), 'batch_size')
    args.dropout = config.getfloat('{}_conf'.format(args.model_name), 'dropout')
    args.weight_decay = config.getfloat('{}_conf'.format(args.model_name), 'weight_decay')
    if 'm' in args.model_name:
        args.rerank_weight = config.getfloat('{}_conf'.format(args.model_name), 'rerank_weight')
        args.class_weight = config.getfloat('{}_conf'.format(args.model_name), 'class_weight')

    if args.parameter_search:
        args.parameter_record = '{}/{}_{}_params.log'.format(RUNNING_PATH, args.model_name, args.dataset_name)
        task_weight_range = np.logspace(-2, 1, num=50, base=10)
        weight_decay_range = np.logspace(-3, -1, num=50, base=10)
        for i in range(args.search_times):
            if args.regularizer_search:
                args.dropout = random.uniform(0.1, 0.6)
                args.weight_decay = random.uniform(0.001, 0.02) if i >= 50 else weight_decay_range[i]
            elif args.mt_search:
                args.rerank_weight = random.uniform(0.01, 10) if i >= 50 else task_weight_range[i]
                args.class_weight = random.uniform(0.01, 10) if i >= 50 else task_weight_range[i]
            logging.info('{}'.format(vars(args)))
            trainer = Trainer(args)
            trainer.run()
            trainer.writer.close()
    else:
        logging.info('{}'.format(vars(args)))
        trainer = Trainer(args)
        trainer.run()
        trainer.writer.close()

if __name__ == '__main__':
    main()
