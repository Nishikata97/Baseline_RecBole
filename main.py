import argparse
import os
from logging import getLogger
from datetime import datetime
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer

from utils import setup_environment, dict2str, set_color
setup_environment()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Baby_Products',
                        choices=['Baby_Products', 'Industrial_and_Scientific', 'Office_Products'], 
                        help='Dataset Name')
    parser.add_argument('--model', type=str, default='SASRec', 
                        help='Sequential Recommendation Model (SASRec, BERT4Rec, SINE, CORE, FEARec, SASRecCPR, TedRec)')
    parser.add_argument('--log_wandb', action='store_true', help='Enable W&B logging')
    args = parser.parse_args()

    configs = ['config/overall.yaml']
    config = Config(model=args.model, dataset=args.dataset, config_file_list=configs)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_name = f"{config['model']}-{config['dataset']}-{timestamp}.pth"
    config['saved_model_file'] = model_name

    init_seed(config['seed'], config['reproducibility'])
    log_dir = os.path.join('./log', config['model'])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    if config['log_wandb']:
        os.makedirs('wandb', exist_ok=True)

    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # Initialize wandb manually before trainer to set custom run name
    if config['log_wandb']:
        import wandb
        run_name = f"{config['model']}_{config['dataset']}_{timestamp}"
        wandb.init(
            project=config['wandb_project'],
            name=run_name,
            config=dict(config.final_config_dict)
        )
        logger.info(f"Initialized W&B with run name: {run_name}")

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color("best valid ", "yellow") + f": \n{dict2str(best_valid_result)}")
    logger.info(set_color("test result", "yellow") + f": \n{dict2str(test_result)}")

if __name__ == '__main__':
    main()
