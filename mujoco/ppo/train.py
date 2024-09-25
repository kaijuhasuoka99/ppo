from ppo.trainer import Trainer
import argparse

def main(args):
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-e', type=str, default='Ant-v4')
    parser.add_argument('--save_weights_folder', type=str, default='./weights/')
    parser.add_argument('--n_worker', '-N', type=int, default=1, help='Number of Worker.')
    parser.add_argument('--horizon', '-T', type=int, default=2048)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--c_1', type=float, default=1)
    parser.add_argument('--c_2', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=3.0e-4)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--iter', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--evaluate_interval', type=int, default=50)
    parser.add_argument('--log_folder', type=str, default='./logs/')

    args = parser.parse_args()
    main(args)