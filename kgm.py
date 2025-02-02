import json
import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from zhipuai import ZhipuAI
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from time import sleep
from tenacity import retry, stop_after_attempt, wait_exponential
import glob
import os

# 配置日志
logging.basicConfig(
    level=logging.ERROR,  # 改为 ERROR 级别以只显示错误信息
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 加载 CSV 数据
try:
    data = pd.read_csv("../mini_size/mini_processed_movies.csv")
except Exception as e:
    logging.error(f"加载 CSV 文件失败：{e}")
    raise

# 数据预处理
try:
    data['text'] = (
        data['NAME'] + "_" + data['ACTORS'] + "_" +
        data['DIRECTORS'] + "_" + data['GENRES'] + "_" +
        data['LANGUAGES'] + "_" +data['REGIONS']+ "_" + data['YEAR'].astype(str)
    )
except Exception as e:
    logging.error(f"数据预处理失败：{e}")
    raise

# 初始化客户端
client = ZhipuAI(api_key="da09b15c5ec54fb1a726faaf14c25c57.9E1jxFYcV9PmjL61")  # 替换为你的API密钥

# ENTITY TYPES
entity_types = {
    "movie": "https://schema.org/Movie",
    "person": "https://schema.org/Person",
    "genre": "https://schema.org/Text",  # Genre is a textual property
    "language": "https://schema.org/Language",
    "releaseYear": "https://schema.org/Date",  # Year as a date entity
    "region": "https://schema.org/Place"  # 添加地区实体类型
}

# RELATION TYPES
relation_types = {
    "hasActor": "https://schema.org/actor",
    "hasDirector": "https://schema.org/director",
    "hasGenre": "https://schema.org/genre",
    "hasLanguage": "https://schema.org/inLanguage",
    "hasReleaseYear": "https://schema.org/datePublished",
    "releasedIn": "https://schema.org/location"
}

# 系统提示
system_prompt = """You are an expert agent specialized in analyzing movie data for knowledge graph construction. Your task is to identify the entities and relations requested in the user prompt from a given movie dataset. You must generate the output in a JSON format containing a list of JSON objects with the following keys: "head", "head_type", "relation", "tail", and "tail_type".

- The "head" key must contain the text of the extracted entity with one of the types from the provided list in the user prompt.
- The "head_type" key must contain the type of the extracted head entity, which must be one of the types from the provided user list.
- The "relation" key must contain the type of relation between the "head" and the "tail".
- The "tail" key must represent the text of an extracted entity which is the tail of the relation.
- The "tail_type" key must contain the type of the tail entity.

The provided entity types include:
- "movie": Represents a movie entity.
- "person": Represents individuals such as actors or directors.
- "genre": Represents the genre of the movie.
- "language": Represents the language of the movie.
- "releaseYear": Represents the year the movie was released.
- "region": Represents the region where the movie was released.

The provided relation types include:
- "hasActor": Defines the relationship between a movie and its actors.
- "hasDirector": Defines the relationship between a movie and its director.
- "hasGenre": Defines the relationship between a movie and its genre.
- "hasLanguage": Defines the relationship between a movie and its language.
- "hasReleaseYear": Defines the relationship between a movie and its release year.
- "releasedIn": Defines the relationship between a movie and its release region.

Attempt to extract as many entities and relations as possible from the input data and follow the specified structure to ensure consistency and completeness.
"""

user_prompt = """Based on the following example, extract entities and relations from the provided text.
Use the following entity types: {entity_types}
Use the following relation types: {relation_types}

Example Input:
NAME: 泰坦尼克号,
ACTORS: 莱昂纳多·迪卡普里奥/凯特·温丝莱特,
DIRECTORS: 詹姆斯·卡梅隆,
GENRES: 剧情/爱情/灾难,
LANGUAGES: 英语,
YEAR: 1997,
REGION: 美国/英国

Example Output:
[
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasActor", "tail": "莱昂纳多·迪卡普里奥", "tail_type": "person"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasActor", "tail": "凯特·温丝莱特", "tail_type": "person"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasDirector", "tail": "詹姆斯·卡梅隆", "tail_type": "person"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasGenre", "tail": "剧情", "tail_type": "genre"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasGenre", "tail": "爱情", "tail_type": "genre"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasGenre", "tail": "灾难", "tail_type": "genre"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasLanguage", "tail": "英语", "tail_type": "language"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasReleaseYear", "tail": "1997", "tail_type": "releaseYear"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "releasedIn", "tail": "美国", "tail_type": "region"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "releasedIn", "tail": "英国", "tail_type": "region"}}
]
--> Beginning of example
# Specification
{specification}
################
# Output
"""

def parse_batch_response(response_content):
    """解析批量响应内容并返回清理后的数据"""
    try:
        parsed_response = json.loads(response_content)
        cleaned_response = []
        if isinstance(parsed_response, list):
            for item in parsed_response:
                cleaned_item = {
                    "head": item["head"].split("/")[-1] if "/" in item["head"] else item["head"],
                    "head_type": item["head_type"],
                    "relation": item["relation"].split("/")[-1] if "/" in item["relation"] else item["relation"],
                    "tail": item["tail"].split("/")[-1] if "/" in item["tail"] else item["tail"],
                    "tail_type": item["tail_type"]
                }
                cleaned_response.append(cleaned_item)
        return cleaned_response
    except json.JSONDecodeError as e:
        logging.error(f"解析响应内容失败: {e}")
        return []

# 提取信息的函数
def extract_information(text, client, model="glm-4-flash"):  # 添加 client 参数
    if pd.isna(text):
        return None
        
    try:
        # 格式化输入文本
        text_parts = text.split('_')
        if len(text_parts) != 7:
            logging.error(f"输入文本格式不正确: {text}")
            return None
            
        # 构建结构化的输入
        movie_data = {
            "NAME": text_parts[0].strip(),
            "ACTORS": text_parts[1].strip(),
            "DIRECTORS": text_parts[2].strip(),
            "GENRES": text_parts[3].strip(),
            "LANGUAGES": text_parts[4].strip(),
            "REGIONS": text_parts[5].strip(),
            "YEAR": text_parts[6].strip()
        }
        
        # 将数据转换为指定格式的字符串
        formatted_spec = (
            f'NAME: {movie_data["NAME"]},\n'
            f'ACTORS: {movie_data["ACTORS"]},\n'
            f'DIRECTORS: {movie_data["DIRECTORS"]},\n'
            f'GENRES: {movie_data["GENRES"]},\n'
            f'LANGUAGES: {movie_data["LANGUAGES"]},\n'
            f'REGIONS: {movie_data["REGIONS"]},\n'
            f'YEAR: {movie_data["YEAR"]}'
        )
        
        # 准备API请求内容
        request_content = user_prompt.format(
            entity_types=json.dumps(entity_types, ensure_ascii=False),
            relation_types=json.dumps(relation_types, ensure_ascii=False),
            specification=formatted_spec
        )
        
        # 使用传入的 client 进行 API 调用
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request_content}
            ],
            temperature=0
        )
        
        # 处理响应
        if hasattr(response, 'choices') and response.choices:
            response_content = response.choices[0].message.content
            
            # 处理返回的响应
            try:
                if response_content.startswith('```json'):
                    response_content = response_content.replace('```json\n', '').replace('\n```', '')
                
                parsed_response = json.loads(response_content)
                if isinstance(parsed_response, list):
                    cleaned_response = []
                    for item in parsed_response:
                        cleaned_item = {
                            "head": item["head"].split("/")[-1] if "/" in item["head"] else item["head"],
                            "head_type": item["head_type"],
                            "relation": item["relation"].split("/")[-1] if "/" in item["relation"] else item["relation"],
                            "tail": item["tail"].split("/")[-1] if "/" in item["tail"] else item["tail"],
                            "tail_type": item["tail_type"]
                        }
                        cleaned_response.append(cleaned_item)
                    return json.dumps(cleaned_response)
                else:
                    logging.error("API返回的数据不是JSON数组格式")
                    return None
            except json.JSONDecodeError as e:
                logging.error(f"API返回的数据不是有效的JSON格式: {e}")
                return None
            
    except Exception as e:
        logging.error(f"调用 OpenAI 接口时出错：{str(e)}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_information_with_retry(text, client, model="glm-4-flash"):
    """添加重试机制的信息提取函数"""
    result = extract_information(text, client, model)  # 传递 client 参数
    if result is None:
        raise Exception("提取失败")
    return result

def save_checkpoint(processed_data, filename="../mini_size/kg_checkpoint.json"):
    """保存检查点"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

def load_checkpoint(filename="../mini_size/kg_checkpoint.json"):
    """加载检查点"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"processed_indices": [], "kg_data": [], "errors": 0}

def process_chunk(args):
    """处理数据块的函数"""
    chunk_data, chunk_id = args
    
    # 每个进程创建自己的API客户端
    client = ZhipuAI(api_key="da09b15c5ec54fb1a726faaf14c25c57.9E1jxFYcV9PmjL61")
    
    checkpoint_file = f"../mini_size/checkpoint_chunk_{chunk_id}.json"
    checkpoint = load_checkpoint(checkpoint_file)
    
    results = checkpoint["kg_data"]
    errors = checkpoint["errors"]
    processed_indices = set(checkpoint["processed_indices"])
    
    # 创建一个有序的待处理索引列表，确保按顺序处理
    remaining_indices = sorted(set(range(len(chunk_data))) - processed_indices)
    
    with tqdm(total=len(chunk_data), desc=f"进程 {chunk_id}", position=chunk_id) as pbar:
        # 更新已处理的进度
        pbar.update(len(processed_indices))
        
        for i in remaining_indices:
            content = chunk_data[i]
            try:
                if pd.isna(content):
                    errors += 1
                    processed_indices.add(i)
                    save_checkpoint({
                        "processed_indices": list(processed_indices),
                        "kg_data": results,
                        "errors": errors
                    }, checkpoint_file)
                    pbar.update(1)
                    continue
                    
                extracted_relations = extract_information_with_retry(content, client)  # 传入client
                
                if extracted_relations:
                    parsed_relations = json.loads(extracted_relations)
                    results.extend(parsed_relations)
                else:
                    errors += 1
                
                processed_indices.add(i)
                save_checkpoint({
                    "processed_indices": list(processed_indices),
                    "kg_data": results,
                    "errors": errors
                }, checkpoint_file)
                
            except Exception as e:
                logging.error(f"处理数据出错: {str(e)}")
                continue
            
            pbar.update(1)
            
    return results, errors

def parallel_process_data(data, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # 确保每个块至少有一条数据
    chunk_size = max(1, len(data) // num_processes)
    
    # 创建数据块，确保所有数据都被分配
    chunks = []
    for i in range(0, len(data), chunk_size):
        end = min(i + chunk_size, len(data))  # 确保不会越界
        chunk_id = i // chunk_size
        chunks.append((data[i:end], chunk_id))
    
    print(f"\n开始并行处理，共 {len(data)} 条数据，分为 {len(chunks)} 个数据块...")
    print(f"每个数据块大约包含 {chunk_size} 条数据")
    
    kg = []
    total_errors = 0
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = list(executor.map(process_chunk, chunks))
        
        for result, errors in futures:
            kg.extend(result)
            total_errors += errors
    
    # 打印处理结果统计
    print(f"\n处理完成：")
    print(f"- 总数据量：{len(data)}")
    print(f"- 成功提取关系数：{len(kg)}")
    print(f"- 处理失败数：{total_errors}")
    
    return kg, total_errors

def batch_extract_information(texts, batch_size=5):
    """批量处理多条数据"""
    if not texts:
        return []
        
    # 构建批量请求的提示
    batch_prompts = []
    for text in texts:
        if pd.isna(text):
            continue
            
        text_parts = text.split('_')
        if len(text_parts) != 6:
            continue
            
        movie_data = {
            "NAME": text_parts[0].strip(),
            "ACTORS": text_parts[1].strip(),
            "DIRECTORS": text_parts[2].strip(),
            "GENRES": text_parts[3].strip(),
            "LANGUAGES": text_parts[4].strip(),
            "REGIONS": text_parts[5].strip(),
            "YEAR": text_parts[6].strip()
        }
        
        formatted_spec = (
            f'Movie {len(batch_prompts) + 1}:\n'
            f'NAME: {movie_data["NAME"]},\n'
            f'ACTORS: {movie_data["ACTORS"]},\n'
            f'DIRECTORS: {movie_data["DIRECTORS"]},\n'
            f'GENRES: {movie_data["GENRES"]},\n'
            f'LANGUAGES: {movie_data["LANGUAGES"]},\n'
            f'REGIONS: {movie_data["REGIONS"]},\n'
            f'YEAR: {movie_data["YEAR"]}\n'
        )
        batch_prompts.append(formatted_spec)
    
    # 合并所有提示
    combined_spec = "\n".join(batch_prompts)
    
    # 发送批量请求
    request_content = user_prompt.format(
        entity_types=json.dumps(entity_types, ensure_ascii=False),
        relation_types=json.dumps(relation_types, ensure_ascii=False),
        specification=combined_spec
    )
    
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request_content}
            ],
            temperature=0
        )
        
        # 处理响应...
        if hasattr(response, 'choices') and response.choices:
            response_content = response.choices[0].message.content
            # 解析和清理响应...
            return parse_batch_response(response_content)
    except Exception as e:
        logging.error(f"批处理请求失败：{e}")
        return []

# 在主处理循环中使用批处理
def process_in_batches(data, batch_size=5):
    kg = []
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    for batch in tqdm(batches, desc="处理数据批次"):
        results = batch_extract_information(batch)
        kg.extend(results)
    
    return kg

if __name__ == "__main__":
    # 删除旧的检查点文件，确保清洁开始
    checkpoint_files = glob.glob("../mini_size/checkpoint_chunk_*.json")
    for f in checkpoint_files:
        os.remove(f)
    
    NUM_PROCESSES = 4
    
    # 加载数据
    data = pd.read_csv("../mini_size/mini_processed_movies.csv")
    total_movies = len(data)
    
    # 数据预处理
    data['text'] = (
        data['NAME'] + "_" + data['ACTORS'] + "_" +
        data['DIRECTORS'] + "_" + data['GENRES'] + "_" +
        data['LANGUAGES'] + "_" + data['REGIONS'] + "_" + 
        data['YEAR'].astype(str)
    )
    
    print(f"开始处理 {total_movies} 条数据...")
    
    # 并行处理数据
    kg, error_count = parallel_process_data(
        data['text'].values,
        num_processes=NUM_PROCESSES
    )
    
    # 保存结果
    if kg:
        kg_relations = pd.DataFrame(kg)
        kg_relations.to_csv("../mini_size/mini_kg_relations.csv", index=False, encoding='utf-8')
        
        # 验证处理是否完整
        processed_movies = len(set(kg_relations['head']))
        print(f"\n验证结果：")
        print(f"- 原始电影数量：{total_movies}")
        print(f"- 处理后电影数量：{processed_movies}")
        print(f"- 提取的关系总数：{len(kg)}")
        print(f"- 处理失败数量：{error_count}")