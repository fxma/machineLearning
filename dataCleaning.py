# 数据清洗 参考 https://www.colabug.com/2711587.html
import pandas as pd
import numpy as np
from functools import reduce

# 读取csv文件到dataframe
df = pd.read_csv('./data/BL-Flickr-Images-Book.csv')

# 不输出省略号
np.set_printoptions(threshold=np.inf)
# pd.set_option('max_colwidth',200)
pd.set_option('display.width',None)
print(df.head(n=6))
#  删除无用列
to_drop = ['Edition Statement',
                  'Corporate Author',
                     'Corporate Contributors',
                     'Former owner',
                     'Engraver',
                     'Contributors',
                     'Issuance type',
                     'Shelfmarks']

df.drop(to_drop, inplace=True, axis=1)
print(df.head())

print(df['Identifier'].is_unique)
# 更改 DataFrame 的索引
df = df.set_index('Identifier')
# df.set_index('Identifier', inplace=True)

# 直接访问每条记录
print(df.loc[206])
print(df.iloc[0])

# 返回数据框数据类型的个数
print(df.get_dtype_counts())

#整理数据中的字段
print(df.loc[1905:, 'Date of Publication'].head(10))
print(df.head(24))
#提取年份
extr = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
print(extr.head())
unwanted_characters = ['[', ',', '-']

#使用apply方法
# def clean_dates(item):
#     dop= str(item.loc['Date of Publication'])
#     if dop == 'nan' or dop[0] == '[':
#         return np.NaN
#
#     for character in unwanted_characters:
#         if character in dop:
#             character_index = dop.find(character)
#             dop = dop[:character_index]
#
#     return dop
#
# df['Date of Publication'] = df.apply(clean_dates, axis = 1)
#alternate way of cleaning Date of Publication
#run cell to see output
unwanted_characters = ['[', ',', '-']

def clean_dates(dop):
    dop = str(dop)
    if dop.startswith('[') or dop == 'nan':
        return 'NaN'
    for character in unwanted_characters:
        if character in dop:
            character_index = dop.find(character)
            dop = dop[:character_index]
    return dop

df['Date of Publication'] = df['Date of Publication'].apply(clean_dates)


def clean_author_names(author):
    author = str(author)
    if author == 'nan':
        return 'NaN'
    author = author.split(',')
    if len(author) == 1:
        name = filter(lambda x: x.isalpha(), author[0])
        return reduce(lambda x, y: x + y, name)
    last_name, first_name = author[0], author[1]
    first_name = first_name[:first_name.find('-')] if '-' in first_name else first_name
    if first_name.endswith(('.', '.|')):
        parts = first_name.split('.')
        if len(parts) > 1:
            first_occurence = first_name.find('.')
            final_occurence = first_name.find('.', first_occurence + 1)
            first_name = first_name[:final_occurence]
        else:
            first_name = first_name[:first_name.find('.')]
    last_name = last_name.capitalize()
    return f'{first_name} {last_name}'

df['Author'] = df['Author'].apply(clean_author_names)

print(df.head())

def clean_title(title):
    if title == 'nan':
        return 'NaN'
    if title[0] == '[':
        title = title[1: title.find(']')]
    if 'by' in title:
        title = title[:title.find('by')]
    elif 'By' in title:
        title = title[:title.find('By')]
    if '[' in title:
        title = title[:title.find('[')]
    title = title[:-2]
    title = list(map(str.capitalize, title.split()))
    return ' '.join(title)

df['Title'] = df['Title'].apply(clean_title)
print(df.head())

# 结合 NumPy 以及 str 方法来清理列
print(df['Place of Publication'].head(10))
print(df.loc[4157862])
print(df.loc[4159587])

pub = df['Place of Publication']
london = pub.str.contains('London')
print(london[:5])

oxford = pub.str.contains('Oxford')
df['Place of Publication'] = np.where(london, 'London',
                                      np.where(oxford, 'Oxford',
                                               pub.str.replace('-', ' ')))
print( df['Place of Publication'].head())
print(df.head())


university_towns = []
with open('./data/university_towns.txt') as file:
    for line in file:
        if '[edit]' in line:
            # Remember this `state` until the next is found
            state = line
        else:
            # Otherwise, we have a city; keep `state` as last-seen
            university_towns.append((state, line))

print(university_towns[:5])

towns_df = pd.DataFrame(university_towns,columns=['State', 'RegionName'])
print(towns_df.head())

def get_citystate(item):
    if ' (' in item:
        return item[:item.find(' (')]
    elif '[' in item:
        return item[:item.find('[')]
    else:
        return item

towns_df =  towns_df.applymap(get_citystate)
print(towns_df.head())


olympics_df = pd.read_csv('./data/olympics.csv')
print(olympics_df.head())

olympics_df = pd.read_csv('./data/olympics.csv', header=1)
print(olympics_df.head())

new_names =  {'Unnamed: 0': 'Country',
              '? Summer': 'Summer Olympics',
              '01 !': 'Gold',
              '02 !': 'Silver',
              '03 !': 'Bronze',
              '? Winter': 'Winter Olympics',
              '01 !.1': 'Gold.1',
              '02 !.1': 'Silver.1',
              '03 !.1': 'Bronze.1',
              '? Games': '# Games',
              '01 !.2': 'Gold.2',
              '02 !.2': 'Silver.2',
              '03 !.2': 'Bronze.2'}

olympics_df.rename(columns=new_names, inplace=True)
print(olympics_df.head())