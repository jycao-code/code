import pandas as pd
import os

def process_ratings_data():
    # 读取 ratings.csv 文件
    df = pd.read_csv('../pre_dataset/ratings.csv')
    
    # 只保留需要的列
    columns_to_keep = ['USER_MD5', 'MOVIE_ID', 'RATING']
    df = df[columns_to_keep]
    
    # 只保留评分大于等于4的数据
    df = df[df['RATING'] >= 4]
    
    # 统计每个用户的高分评价次数
    user_rating_counts = df['USER_MD5'].value_counts()
    # 筛选出评价次数大于等于10次的用户
    qualified_users = user_rating_counts[user_rating_counts >=50].index
    # 只保留这些用户的评分数据
    df = df[df['USER_MD5'].isin(qualified_users)]
    
    # 统计每部电影的高分评价次数
    movie_rating_counts = df['MOVIE_ID'].value_counts()
    # 筛选出被评价次数大于等于10次的电影
    qualified_movies = movie_rating_counts[movie_rating_counts >=50].index
    # 只保留这些电影的评分数据
    df = df[df['MOVIE_ID'].isin(qualified_movies)]
    
    # 保存处理后的数据到新文件
    output_path = os.path.join('../mini_size', 'mini_processed_ratings.csv')
    df.to_csv(output_path, index=False)
    
    print(f"评分数据处理完成。处理后的数据保存在: {output_path}")
    print(f"处理后的评分数据集包含 {len(df)} 条记录")
    print(f"符合条件的用户数量: {len(qualified_users)}")
    print(f"符合条件的电影数量: {len(qualified_movies)}")
    
    return set(qualified_movies)

def process_movies_data(valid_movie_ids):
    # 读取 movies.csv 文件
    df = pd.read_csv('../pre_dataset/movies.csv')
    
    # 只保留需要的列
    columns_to_keep = ['MOVIE_ID', 'NAME', 'ACTORS', 'DIRECTORS', 'GENRES', 'LANGUAGES', 'YEAR', 'REGIONS']
    df = df[columns_to_keep]
    
    # 删除任何包含空值的行
    df = df.dropna()
    
    # 去除重复行
    df = df.drop_duplicates()
    
    # 只保留在评分数据中出现的电影
    df = df[df['MOVIE_ID'].isin(valid_movie_ids)]
    
    # 确保 dataset 文件夹存在
    os.makedirs('../dataset', exist_ok=True)
    
    # 保存处理后的数据到新文件
    output_path = os.path.join('../mini_size', 'mini_processed_movies.csv')
    df.to_csv(output_path, index=False)
    
    print(f"电影数据处理完成。处理后的数据保存在: {output_path}")
    print(f"处理后的电影数据集包含 {len(df)} 条记录")
    
    # 返回处理后的电影ID集合
    return set(df['MOVIE_ID'])

if __name__ == "__main__":
    # 先处理评分数据，获取高分高频电影ID集合
    valid_movie_ids = process_ratings_data()
    # 处理电影数据，只保留有效的电影
    final_movie_ids = process_movies_data(valid_movie_ids)
    # 再次处理评分数据，只保留最终的电影
    df_ratings = pd.read_csv('../mini_size/mini_processed_ratings.csv')
    df_ratings = df_ratings[df_ratings['MOVIE_ID'].isin(final_movie_ids)]
    df_ratings.to_csv('../mini_size/mini_processed_ratings.csv', index=False)
    print(f"最终的评分数据集包含 {len(df_ratings)} 条记录")

