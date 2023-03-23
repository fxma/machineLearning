'''
Bar Scatters Charts（水平条形图）
'''
# 加载numpy和pandas包
import numpy as np
import pandas as pd

# 加载plotly包
import plotly as py
from plotly.offline import plot


# 加载数据
timesData = pd.read_csv("./timesData.csv")

# import graph objects as "go" and import tools
import plotly.graph_objs as go
from plotly import tools
from plotly import subplots
import matplotlib.pyplot as plt
# 准备 data frames
df2016 = timesData[timesData.year == 2016].iloc[:7,:]

y_saving = [each for each in df2016.research]
y_net_worth  = [float(each) for each in df2016.income]
x_saving = [each for each in df2016.university_name]
x_net_worth  = [each for each in df2016.university_name]
trace0 = go.Bar(
                x=y_saving,
                y=x_saving,
                marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),
                name='research',
                orientation='h',    # Bar的方向设置为水平（horizontal）
)
trace1 = go.Scatter(
                x=y_net_worth,
                y=x_net_worth,
                mode='lines+markers',
                line=dict(color='rgb(63, 72, 204)'),
                name='income',
)
layout = dict(
                title='Citations and income',
                yaxis=dict(showticklabels=True,domain=[0, 0.85]),  # showticklables用来决定是否显示每个bar的旁注，domain用来设置y轴长度
                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),
                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42]),
                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25),
                legend=dict(x=0.029,y=1.038,font=dict(size=10) ),  #设置图例标志的大小和位置
                margin=dict(l=200, r=20,t=70,b=70), # 设置bar旁注的长度、大小等
                paper_bgcolor='rgb(248, 248, 255)', # 设置整个面板的背景色
                plot_bgcolor='rgb(248, 248, 255)',  # 设置图像部份的背景色
)
annotations = []
y_s = np.round(y_saving, decimals=2)
y_nw = np.rint(y_net_worth)
# 添加 labels
for ydn, yd, xd in zip(y_nw, y_s, x_saving):
    # 标注散点图
    annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 4,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,color='rgb(63, 72, 204)'),showarrow=False))
    # 标注条形图
    annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3,text=str(yd),font=dict(family='Arial', size=12,color='rgb(171, 50, 96)'),showarrow=False))

layout['annotations'] = annotations

# 创建两个子图
# fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
#                           shared_yaxes=False, vertical_spacing=0.001)
fig = subplots.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                          shared_yaxes=False, vertical_spacing=0.001)

fig.add_trace(trace0, 1, 1)
fig.add_trace(trace1, 1, 2)

fig['layout'].update(layout)
plot(fig)