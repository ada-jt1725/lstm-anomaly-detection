import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 512
lr = 1e-6

best_loss = 1000000000000.0
latent_space_dim = 20
train_test_ratio = 1
slide_range = 300
signal_columns = ['流入速率(Mbps)', '流出速率(Mbps)']
n_features = 2
base_path = 'exchange-example-delete-error.csv'
base_path_2 = 'in_out.xlsx'
# base_path = 'exchange-2_cpc_results.csv'
train_dataset_path = 'train_dataset.csv'
after_train_dataset_path = 'after_train_dataset.csv'
test_dataset_path = 'test_dataset.csv'
after_test_dataset_path = 'after_test_dataset.csv'
anomaly_dataset_path = 'anomaly_dataset_binary.csv'
anomaly_dataset_path_ori = 'anomaly_dataset.csv'
encoder_path = 'models/encoder.pt'
decoder_path = 'models/decoder.pt'
critic_x_path = 'models/critic_x.pt'
critic_z_path = 'models/critic_z.pt'
MODEL_PATH = 'model.pth'
loss_dataset_path = 'loss.csv'
accu_dataset_path = 'accuracy.csv'
n_epochs=400