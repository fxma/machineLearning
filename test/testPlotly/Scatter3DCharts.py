'''
3D Scatter Plot (3D散点图)
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
# 准备数据
dataframe = timesData[timesData.year == 2015]

# 构造3D散点图trace1
trace1 = go.Scatter3d(
    x=dataframe.world_rank,
    y=dataframe.research,
    z=dataframe.citations,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(72,61,139)',  # RGB颜色对照表可参考：https://tool.oschina.net/commons?type=3
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )

)
fig = go.Figure(data=data, layout=layout)
plot(fig)
