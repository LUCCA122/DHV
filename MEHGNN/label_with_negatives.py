# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:58:38 2024

@author: 10176
"""

import pandas as pd
import numpy as np

# 步骤1: 读取正样本数据
df = pd.read_csv('data/raw/VULKG/labels.csv')

# 提取所有的CVE编号和Product Version
all_vulnerabilities = df['cveID'].unique()
all_product_versions = df['Product Version'].unique()

# 步骤2: 生成所有可能的组合
all_combinations = pd.MultiIndex.from_product([all_vulnerabilities, all_product_versions], names=['cveID', 'Product Version']).to_frame(index=False)

# 步骤3: 筛选出不在原始数据中的组合作为负样本候选
negative_samples_candidates = pd.concat([all_combinations, df[['cveID', 'Product Version']], df[['cveID', 'Product Version']]]).drop_duplicates(keep=False)

# 步骤4: 随机选择负样本
negative_samples = negative_samples_candidates.sample(n=min(50000, len(negative_samples_candidates)))

# 给负样本标记为0
negative_samples['label'] = 0

# 将负样本添加到原始数据中
df_with_negatives = pd.concat([df, negative_samples])

# 步骤5: 保存到新的Excel文件
df_with_negatives.to_excel('data/raw/VULKG/label_with_negatives.xlsx', index=False)