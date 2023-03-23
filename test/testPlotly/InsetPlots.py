'''
Inset Plots（内置图）
'''
# 加载numpy和pandas包
import numpy as np
import pandas as pd

# 加载plotly包
import plotly as py
from plotly.offline import plot


# 加载数据
timesData = pd.read_csv("./timesData.csv")

import plotly.graph_objs as go

# import figure factory
import plotly.figure_factory as ff
# 准备数据
dataframe = timesData[timesData.year == 2015]
# 第一个线型图
trace1 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.teaching,
    name = "teaching",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)
# 第二个线型图
trace2 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.income,
    xaxis='x2',
    yaxis='y2',
    name = "income",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2',
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = 'Income and Teaching vs World Rank of Universities'

)

fig = go.Figure(data=data, layout=layout)
plot(fig)