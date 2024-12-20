import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np
from utils import weights_init
from model import CNNEncoder
from model import RelationNetwork
from omniglot import *
import argparse

if __name__ == '__main__':

    print("aa")

    parser = argparse.ArgumentParser(description="One Shot Visual Recognition")

    # 如果需要，修改默认值、类型或描述
    parser.add_argument("-f", "--feature_dim", type=int, default=64, help="Dimension of feature space.")
    parser.add_argument("-r", "--relation_dim", type=int, default=8, help="Dimension of relation space.")
    parser.add_argument("-w", "--class_num", type=int, default=5, help="Number of classes in one episode.")
    parser.add_argument("-s", "--sample_num_per_class", type=int, default=1, help="Number of samples per class.")
    parser.add_argument("-b", "--batch_num_per_class", type=int, default=19, help="Number of batches per class.")
    parser.add_argument("-e", "--episode", type=int, default=10, help="Number of training episodes.")
    parser.add_argument("-t", "--test_episode", type=int, default=1000, help="Number of testing episodes.")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device ID.")
    parser.add_argument("-u", "--hidden_unit", type=int, default=10, help="Number of hidden units.")
    parser.add_argument("-d", "--data_path", type=str, default="./data/omniglot_resized", help="Path to the dataset.")

    args = parser.parse_args()

    # Hyper Parameters
    FEATURE_DIM = args.feature_dim
    RELATION_DIM = args.relation_dim
    CLASS_NUM = args.class_num
    SAMPLE_NUM_PER_CLASS = args.sample_num_per_class  # 注意这里与上面定义的默认值不一致
    BATCH_NUM_PER_CLASS = args.batch_num_per_class
    EPISODE = args.episode
    TEST_EPISODE = args.test_episode
    LEARNING_RATE = args.learning_rate
    GPU = args.gpu
    HIDDEN_UNIT = args.hidden_unit
    DATA_PATH = args.data_path

    # 数据预处理
    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])


    # 模型加载函数
    def load_model(model, model_path: str) -> None:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=f'cuda:{GPU}'))
            print(f"Load {model.__class__.__name__} success")
        else:
            print(f"{model_path} does not exist")


    # 模型保存函数
    def save_model(model, model_name: str) -> None:
        torch.save(model.state_dict(), model_name)
        print(f"Save {model.__class__.__name__} success")


    # 初始化数据集
    print("init data folders")
    metatrain_character_folders, metatest_character_folders = omniglot_character_folders(DATA_PATH)

    # 初始化神经网络
    print("init neural networks")
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    # 确保GPU可用
    if GPU >= 0 and torch.cuda.is_available():
        feature_encoder.cuda(GPU)
        relation_network.cuda(GPU)

    # 优化器设置
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    # 加载模型
    load_model(feature_encoder, f"./models/omniglot_feature_encoder_{CLASS_NUM}way_{SAMPLE_NUM_PER_CLASS}shot.pkl")
    load_model(relation_network, f"./models/omniglot_relation_network_{CLASS_NUM}way_{SAMPLE_NUM_PER_CLASS}shot.pkl")

    # 训练过程
    print("Training...")
    last_accuracy = 0.0

    for episode in range(EPISODE):

        # 数据加载
        degrees = random.choice([0, 90, 180, 270])
        task = OmniglotTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        sample_dataloader = get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False,
                                            rotation=degrees)
        batch_dataloader = get_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS, split="test", shuffle=True,
                                           rotation=degrees)

        # 获取支持集和查询集
        try:
            samples, sample_labels = next(iter(sample_dataloader))
            batches, batch_labels = next(iter(batch_dataloader))
        except StopIteration:
            print(f"Error in DataLoader at episode {episode}. Not enough data.")
            continue

        # 特征提取
        with torch.no_grad():
            sample_features = feature_encoder(Variable(samples).cuda(GPU))
            batch_features = feature_encoder(Variable(batches).cuda(GPU))

        # 特征扩展和关系网络
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 5, 5)
        relations = relation_network(relation_pairs).view(-1, CLASS_NUM)

        # 损失计算和优化
        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(
            torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1, 1), 1)).cuda(GPU)
        loss = mse(relations, one_hot_labels)

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        # 反向传播计算梯度
        loss.backward()

        # 更新权重
        feature_encoder_optim.step()
        relation_network_optim.step()

        # 学习率调度器更新
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # 打印训练信息
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1} | Loss: {loss.item()}")

        # 测试模型准确性
        if (episode + 1) % 100 == 0:
            total_rewards = 0

            for _ in range(TEST_EPISODE):
                degrees = random.choice([0, 90, 180, 270])
                task = OmniglotTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
                sample_dataloader = get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                    shuffle=False,
                                                    rotation=degrees)
                test_dataloader = get_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS, split="test", shuffle=True,
                                                  rotation=degrees)

                sample_images, sample_labels = next(iter(sample_dataloader))
                test_images, test_labels = next(iter(test_dataloader))

                # 测试时关闭梯度计算
                with torch.no_grad():
                    sample_features = feature_encoder(Variable(sample_images).cuda(GPU))
                    test_features = feature_encoder(Variable(test_images).cuda(GPU))

                    sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1,
                                                                              1)
                    test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
                    test_features_ext = torch.transpose(test_features_ext, 0, 1)

                    relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 5,
                                                                                                 5)
                    relations = relation_network(relation_pairs).view(-1, CLASS_NUM)

                    _, predict_labels = torch.max(relations.data, 1)
                    rewards = [1 if predict_labels[j].cpu() == test_labels[j] else 0 for j in
                               range(CLASS_NUM * BATCH_NUM_PER_CLASS)]

                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards / (1.0 * CLASS_NUM * BATCH_NUM_PER_CLASS * TEST_EPISODE)
            print(f"Test Accuracy: {test_accuracy}")

            # 保存模型
            if test_accuracy > last_accuracy:
                save_model(feature_encoder,
                           f"./models/omniglot_feature_encoder_{CLASS_NUM}way_{SAMPLE_NUM_PER_CLASS}shot.pkl")
                save_model(relation_network,
                           f"./models/omniglot_relation_network_{CLASS_NUM}way_{SAMPLE_NUM_PER_CLASS}shot.pkl")
                last_accuracy = test_accuracy