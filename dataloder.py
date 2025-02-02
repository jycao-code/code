import pandas as pd
import os
import random
import numpy as np

def split_train_valid_test(train_file, output_dir, train_ratio=0.8, valid_ratio=0.1):
    """
    将交互数据划分为训练集、验证集和测试集
    
    Args:
        train_file: 原始训练数据文件路径
        output_dir: 输出目录路径
        train_ratio: 训练集比例
        valid_ratio: 验证集比例(从训练集中划分)
    """
    # 读取原始训练数据
    train_data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            user_items = line.strip().split()
            user = user_items[0]
            items = user_items[1:]
            train_data.append((user, items))
    
    # 为每个用户划分数据
    train_set = []
    valid_set = []
    test_set = []
    
    for user, items in train_data:
        # 随机打乱物品列表
        items = list(items)
        random.shuffle(items)
        
        # 计算划分点
        n_items = len(items)
        n_train = int(n_items * train_ratio)
        n_valid = int(n_train * valid_ratio)
        
        # 划分数据
        train_items = items[:n_train-n_valid]
        valid_items = items[n_train-n_valid:n_train]
        test_items = items[n_train:]
        
        # 只有当有交互记录时才添加到相应集合
        if train_items:
            train_set.append(f"{user} {' '.join(train_items)}")
        if valid_items:
            valid_set.append(f"{user} {' '.join(valid_items)}")
        if test_items:
            test_set.append(f"{user} {' '.join(test_items)}")
    
    # 保存划分后的数据集
    with open(os.path.join(output_dir, 'train1.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_set))
    
    with open(os.path.join(output_dir, 'valid.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_set))
    
    with open(os.path.join(output_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_set))
    
    return len(train_set), len(valid_set), len(test_set)

def create_entity_mapping(kg_file, movie_mapping, movies_df, output_dir):
    """创建实体映射"""
    kg_df = pd.read_csv(kg_file)
    
    # 创建电影名称到MOVIE_ID的映射
    movie_name_to_id = dict(zip(movies_df['NAME'], movies_df['MOVIE_ID']))
    movie_id_to_name = dict(zip(movies_df['MOVIE_ID'], movies_df['NAME']))
    
    # 创建MOVIE_ID到remap_id的映射
    movie_id_to_remap = dict(zip(movie_mapping['org_id'], movie_mapping['remap_id']))
    
    # 获取所有实体
    head_entities = set(kg_df['head'].unique())
    tail_entities = set(kg_df['tail'].unique())
    all_entities = sorted(head_entities.union(tail_entities))
    
    # 创建实体映射
    entity_mapping = []
    next_id = 0
    
    # 先处理电影实体
    movie_entities = []
    for entity in all_entities:
        if entity in movie_name_to_id:  # 如果是电影名称
            movie_id = movie_name_to_id[entity]
            if movie_id in movie_id_to_remap:  # 如果这个电影ID在映射中
                remap_id = movie_id_to_remap[movie_id]
                entity_mapping.append((movie_id, remap_id))
                movie_entities.append(entity)
                next_id = max(next_id, remap_id + 1)
    
    # 处理其他实体（非电影实体）
    other_entities = sorted(set([e for e in all_entities if e not in movie_entities]))
    for entity in other_entities:
        entity_mapping.append((entity, next_id))
        next_id += 1
    
    # 转换为DataFrame并保存
    entity_df = pd.DataFrame(entity_mapping, columns=['org_id', 'remap_id'])
    entity_df.to_csv(os.path.join(output_dir, 'entity_list.txt'), sep=' ', index=False)
    
    return entity_df

def create_kg_final(kg_file, entity_mapping, relation_mapping, movies_df, output_dir):
    """创建知识图谱的最终映射文件"""
    kg_df = pd.read_csv(kg_file)
    
    # 创建电影名称到MOVIE_ID的映射
    movie_name_to_id = dict(zip(movies_df['NAME'], movies_df['MOVIE_ID']))
    
    # 创建映射字典
    entity_dict = dict(zip(entity_mapping['org_id'], entity_mapping['remap_id']))
    relation_dict = dict(zip(relation_mapping['org_id'], relation_mapping['remap_id']))
    
    # 转换头实体、关系和尾实体的ID
    kg_final = []
    for _, row in kg_df.iterrows():
        try:
            # 如果是电影名称，转换为MOVIE_ID
            head = movie_name_to_id[row['head']] if row['head'] in movie_name_to_id else row['head']
            tail = row['tail']  # 尾实体保持原样
            
            head_id = entity_dict[head]
            relation_id = relation_dict[row['relation']]
            tail_id = entity_dict[tail]
            kg_final.append(f"{head_id} {relation_id} {tail_id}")
        except KeyError as e:
            print(f"Warning: 实体未找到: {e}")
            continue
    
    # 保存结果
    with open(os.path.join(output_dir, 'kg_final.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(kg_final))
    
    return len(kg_final)

def create_id_mappings(rating_file, kg_file, movies_file, output_dir):
    """
    创建用户、电影和关系的ID映射文件，并生成训练数据格式
    
    Args:
        rating_file: 评分数据文件路径
        kg_file: 知识图谱关系文件路径
        movies_file: 电影信息文件路径
        output_dir: 输出目录路径
    """
    # 读取评分数据、关系数据和电影数据
    rating_df = pd.read_csv(rating_file)
    kg_df = pd.read_csv(kg_file)
    movies_df = pd.read_csv(movies_file)
    
    # 获取唯一的用户ID、电影ID和关系类型
    user_ids = sorted(rating_df['USER_MD5'].unique())
    movie_ids = sorted(rating_df['MOVIE_ID'].unique())
    relation_types = sorted(kg_df['relation'].unique())
    
    # 创建用户ID映射
    user_mapping = pd.DataFrame({
        'org_id': user_ids,
        'remap_id': range(len(user_ids))
    })
    
    # 创建电影ID映射
    movie_mapping = pd.DataFrame({
        'NAME': [movies_df.loc[movies_df['MOVIE_ID'] == mid, 'NAME'].iloc[0] if mid in movies_df['MOVIE_ID'].values else mid for mid in movie_ids],
        'org_id': movie_ids,
        'remap_id': range(len(movie_ids))
    })
    
    # 创建关系类型映射
    relation_mapping = pd.DataFrame({
        'org_id': relation_types,
        'remap_id': range(len(relation_types))
    })
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存映射文件
    user_mapping.to_csv(os.path.join(output_dir, 'user_list.txt'), 
                       sep=' ', index=False)
    movie_mapping.to_csv(os.path.join(output_dir, 'item_list.txt'),
                        sep=' ', index=False)
    relation_mapping.to_csv(os.path.join(output_dir, 'relation_list.txt'),
                          sep=' ', index=False)
    
    # 创建用户-电影交互数据
    # 将原始ID转换为映射后的ID
    rating_df['user_idx'] = rating_df['USER_MD5'].map(dict(zip(user_mapping['org_id'], user_mapping['remap_id'])))
    rating_df['movie_idx'] = rating_df['MOVIE_ID'].map(dict(zip(movie_mapping['org_id'], movie_mapping['remap_id'])))
    
    # 按用户ID分组并整理交互数据
    train_data = []
    for user_idx, group in rating_df.groupby('user_idx'):
        # 获取该用户评分过的所有电影ID
        movie_indices = group['movie_idx'].tolist()
        # 将用户ID和电影ID列表组合成一行
        train_line = [str(user_idx)] + [str(idx) for idx in movie_indices]
        train_data.append(' '.join(train_line))
    
    # 保存训练数据
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_data))
    
    # 在生成完train.txt后，调用划分函数
    train_size, valid_size, test_size = split_train_valid_test(
        os.path.join(output_dir, 'train.txt'),
        output_dir
    )
    
    print(f"数据集划分完成：")
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {valid_size}")
    print(f"测试集大小: {test_size}")
    
    # 创建实体映射
    entity_mapping = create_entity_mapping(kg_file, movie_mapping, movies_df, output_dir)
    print(f"已创建实体映射文件，包含 {len(entity_mapping)} 个实体")
    
    # 创建知识图谱最终映射
    n_relations = create_kg_final(kg_file, entity_mapping, relation_mapping, movies_df, output_dir)
    print(f"已创建知识图谱映射文件，包含 {n_relations} 个关系")
    
    return user_mapping, movie_mapping, relation_mapping, entity_mapping

if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    random.seed(42)
    np.random.seed(42)
    
    # 设置文件路径
    rating_file = "../mini_size/mini_processed_ratings.csv"
    kg_file = "../mini_size/mini_kg_relations.csv"
    movies_file = "../mini_size/mini_processed_movies.csv"
    output_dir = "../dataset"
    
    # 创建映射并划分数据集
    user_map, movie_map, relation_map, entity_map = create_id_mappings(
        rating_file, kg_file, movies_file, output_dir)
    
    print(f"已创建用户映射文件，包含 {len(user_map)} 个用户")
    print(f"已创建电影映射文件，包含 {len(movie_map)} 个电影")
    print(f"已创建关系映射文件，包含 {len(relation_map)} 个关系类型")
