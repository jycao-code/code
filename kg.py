import json
import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from zhipuai import ZhipuAI

# 配置日志
logging.basicConfig(
    level=logging.ERROR,  # 改为 ERROR 级别以只显示错误信息
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 加载 CSV 数据
try:
    data = pd.read_csv("../dataset/processed_movies.csv")
except Exception as e:
    logging.error(f"加载 CSV 文件失败：{e}")
    raise

# 数据预处理
try:
    data['text'] = (
        data['NAME'] + "_" + data['ACTORS'] + "_" +
        data['DIRECTORS'] + "_" + data['GENRES'] + "_" +
        data['LANGUAGES'] + "_" + data['YEAR'].astype(str)
    )
except Exception as e:
    logging.error(f"数据预处理失败：{e}")
    raise

# 初始化客户端
client = ZhipuAI(api_key="da09b15c5ec54fb1a726faaf14c25c57.9E1jxFYcV9PmjL61")  

# ENTITY TYPES
entity_types = {
    "movie": "https://schema.org/Movie",
    "person": "https://schema.org/Person",
    "genre": "https://schema.org/Text",  # Genre is a textual property
    "language": "https://schema.org/Language",
    "releaseYear": "https://schema.org/Date",  # Year as a date entity
}

# RELATION TYPES
relation_types = {
    "hasActor": "https://schema.org/actor",
    "hasDirector": "https://schema.org/director",
    "hasGenre": "https://schema.org/genre",
    "hasLanguage": "https://schema.org/inLanguage",
    "hasReleaseYear": "https://schema.org/datePublished",
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

The provided relation types include:
- "hasActor": Defines the relationship between a movie and its actors.
- "hasDirector": Defines the relationship between a movie and its director.
- "hasGenre": Defines the relationship between a movie and its genre.
- "hasLanguage": Defines the relationship between a movie and its language.
- "hasReleaseYear": Defines the relationship between a movie and its release year.

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
YEAR: 1997

Example Output:
[
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasActor", "tail": "莱昂纳多·迪卡普里奥", "tail_type": "person"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasActor", "tail": "凯特·温丝莱特", "tail_type": "person"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasDirector", "tail": "詹姆斯·卡梅隆", "tail_type": "person"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasGenre", "tail": "剧情", "tail_type": "genre"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasGenre", "tail": "爱情", "tail_type": "genre"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasGenre", "tail": "灾难", "tail_type": "genre"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasLanguage", "tail": "英语", "tail_type": "language"}},
    {{"head": "泰坦尼克号", "head_type": "movie", "relation": "hasReleaseYear", "tail": "1997", "tail_type": "releaseYear"}}
]

--> Beginning of example
# Specification
{specification}
################
# Output
"""

# 提取信息的函数
def extract_information(text, model="glm-4"):  # 使用GLM-4模型
    if pd.isna(text):
        return None
        
    try:
        # 格式化输入文本
        text_parts = text.split('_')
        if len(text_parts) != 6:
            logging.error(f"输入文本格式不正确: {text}")
            return None
            
        # 构建结构化的输入
        movie_data = {
            "NAME": text_parts[0].strip(),
            "ACTORS": text_parts[1].strip(),
            "DIRECTORS": text_parts[2].strip(),
            "GENRES": text_parts[3].strip(),
            "LANGUAGES": text_parts[4].strip(),
            "YEAR": text_parts[5].strip()
        }
        
        # 将数据转换为指定格式的字符串
        formatted_spec = (
            f'NAME: {movie_data["NAME"]},\n'
            f'ACTORS: {movie_data["ACTORS"]},\n'
            f'DIRECTORS: {movie_data["DIRECTORS"]},\n'
            f'GENRES: {movie_data["GENRES"]},\n'
            f'LANGUAGES: {movie_data["LANGUAGES"]},\n'
            f'YEAR: {movie_data["YEAR"]}'
        )
        
        # 准备API请求内容
        request_content = user_prompt.format(
            entity_types=json.dumps(entity_types, ensure_ascii=False),
            relation_types=json.dumps(relation_types, ensure_ascii=False),
            specification=formatted_spec
        )
        
        # 使用客户端实例进行调用
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

# 知识图谱数据提取
kg = []
processed_count = 0
error_count = 0
# 处理所有数据
total_records = len(data['text'].values)
for i, content in tqdm(enumerate(data['text'].values),
                      total=total_records,
                      desc="处理电影数据",
                      ncols=100):
    try:
        if pd.isna(content):
            error_count += 1
            continue
            
        extracted_relations = extract_information(content)
        
        if extracted_relations:
            try:
                parsed_relations = json.loads(extracted_relations)
                kg.extend(parsed_relations)
                processed_count += 1
            except json.JSONDecodeError as e:
                error_count += 1
                logging.error(f"第 {i + 1} 条数据的 JSON 解析失败：{e}")
        else:
            error_count += 1
            
    except Exception as e:
        error_count += 1
        logging.error(f"处理第 {i + 1} 条数据时发生意外错误：{e}")

# 保存知识图谱关系
if kg:
    try:
        kg_relations = pd.DataFrame(kg)
        kg_relations.to_csv("../dataset/kg_relations.csv", index=False, encoding='utf-8')
        print(f"成功将知识图谱关系保存到 'kg_relations.csv'，共 {len(kg)} 条关系")
    except Exception as e:
        logging.error(f"保存知识图谱关系时失败：{e}")
else:
    logging.error("没有提取到有效的知识图谱数据，未生成文件")

"""
# 可视化知识图谱
G = nx.Graph()
for _, row in kg_relations.iterrows():
    G.add_edge(row['head'], row['tail'], label=row['relation'])
pos = nx.spring_layout(G, seed=47, k=0.9)
labels = nx.get_edge_attributes(G, 'label')
plt.figure(figsize=(15, 15))
nx.draw(G, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, label_pos=0.3, verticalalignment='baseline')
plt.title('Product Knowledge Graph')
plt.show()
"""