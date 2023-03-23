'''
Line Charts（折线图）
'''
# 加载numpy和pandas包
import numpy as np
import pandas as pd

# 加载plotly包
# import plotly.plotly as py
import plotly as py
from plotly.offline import plot
# from plotly.offline import init_notebook_mode, plot
# init_notebook_mode(connected=True)
import plotly.graph_objs as go

# 云词库
from wordcloud import WordCloud

# 加载matplotlib包
import matplotlib.pyplot as plt

# 加载数据
timesData = pd.read_csv("./timesData.csv")

# timesData相关信息
timesData.info()
# 头10行
timesData.head(10)

# 创建 data frame
df = timesData.iloc[:100,:]

# import graph objects as "go"
import plotly.graph_objs as go

# 设置第一条折线trace1
# go.Scatter可以创建一个散点图或者折线图的对象，我们将其命名为trace1
trace1 = go.Scatter(
                    x = df.world_rank,   # 定义坐标轴的映射关系，将world_rank这一列作为x轴
                    y = df.citations,    # 同理，将citations这一列作为y轴
                    mode = "lines",      # 我们要绘制折线图，所以将mode设置为“lines”
                    name = "citations",  # 将这条折线命名为citations
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    # maker用来定义点的性质，如颜色、大小等
                    text= df.university_name)
                    # 将university_name一列设置为悬停文本（鼠标悬停在图片上方显示的内容）

# 设置第二条折线trace2
trace2 = go.Scatter(
                    x = df.world_rank,
                    y = df.teaching,
                    mode = "lines+markers", #绘制的折线图由散点连接而成，即lines+markers
                    name = "teaching",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df.university_name)

data = [trace1, trace2]

# 添加图层layout
layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
              # 设置图像的标题
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
              # 设置x轴名称，x轴刻度线的长度，不显示零线
             )

# 将data与layout组合为一个图像
fig = dict(data = data, layout = layout)
# 绘制图像
plot(fig)